from torch import nn
import argparse
import torch
from torch import Tensor as T
from torch.nn.utils import rnn
from data_utils.model_utils import model_builder, load_pretrained_model
from utils.gpu_utils import get_single_free_gpu
from torch.autograd import Variable
from tqdm import tqdm
from envs import OUTPUT_FOLDER
from os.path import join
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from collections import Counter
from data_utils.findcat_fast import contains_subsequence

def find_subsequence(target: list, sequence: list):
    if len(target) == 0:
        return -1
    seq_len = len(sequence)
    matched = 0
    remaining = sequence
    start_idx = 0
    for t in target:
        idx = start_idx
        while idx < len(sequence) and sequence[idx] != t:
            idx = idx + 1
        if idx >= len(remaining):
            return seq_len
        else:
            matched = matched + 1
            if matched == len(target):
                return idx
            else:
                start_idx = idx + 1


def find_subsequence_unorder(target: list, sequence: list):
    if len(target) == 0:
        return len(sequence)
    seq_len = len(sequence)
    target_len = len(target)
    target_set = set(target)
    for i in range(target_len, seq_len+1):
        sub_seq = set(sequence[:i])
        if target_set.issubset(sub_seq):
            return i


ORIG_PROBE_MODEL_NAME = join(OUTPUT_FOLDER, 'train_fastsingle_cat_20000_42_300_0.5.pkl.gz.models', 'model_False_0.0_365_168_dev_0.9797.pkl')
DROP_PROBE_MODEL_NAME = join(OUTPUT_FOLDER, 'train_fastsingle_cat_20000_42_300_0.5.pkl.gz.models', 'model_True_0.1_488_169_dev_0.9992.pkl')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

class LSTMWrapper(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layer:int, concat=False, bidir=True, dropout=0.3, return_last=True):
        """
        :param input_dim:
        :param hidden_dim:
        :param n_layer:
        :param concat:
        :param bidir: bi-direction
        :param dropout:
        :param return_last:
        """
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input: T, input_lengths: T=None):
        # input_length must be in decreasing order if input_lengths is not none
        bsz, slen = input.shape[0], input.shape[1]
        output = input
        outputs = []
        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)
            if input_lengths is not None:
                lens = input_lengths.data.cpu().numpy()
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, _ = self.rnns[i](output)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class ProberModel(nn.Module):
    def __init__(self, config):
        super(ProberModel, self).__init__()
        self.config = config
        self.model = model_builder(args=self.config)
        if self.config.pre_trained_file_name is not None:
            self.model.load_state_dict(torch.load(self.config.pre_trained_file_name))
            print('Loading model from {}'.format(self.config.pre_trained_file_name))
        # self.model.bert.register_forward_hook(get_activation('bert'))
        self.model.bert.encoder.register_forward_hook(get_activation('encoder'))
        for param in self.model.parameters():
            param.requires_grad = False

        self.lstm_encoder = LSTMWrapper(input_dim=self.config.hidden_dim,
                                        hidden_dim=self.config.lstm_hidden_dim,
                                        n_layer=self.config.lstm_layers)
        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.classifier = nn.Linear(2 * self.config.lstm_hidden_dim, self.config.num_labels)

    def forward(self, input, attn_mask, labels, label_mask):
        self.model(input, attn_mask)
        # bert_output = activation['bert']
        bert_output = activation['encoder']
        # print(len(bert_output))
        # print(bert_output[0].shape)
        seq_output = bert_output[0]
        seq_output = self.lstm_encoder(seq_output)
        seq_output = self.dropout(seq_output)
        seq_scores = self.classifier(seq_output)
        if self.config.loss_type == 'bce':
            loss = loss_computation(scores=seq_scores, labels=labels, label_mask=label_mask)
        else:
            # loss = adversarial_loss_computation(scores=seq_scores, labels=labels, label_mask=label_mask)
            loss = rank_loss_computation(scores=seq_scores, labels=labels, label_mask=label_mask)
        return loss, seq_scores

def loss_computation(scores, labels, label_mask):
    scores[label_mask == 0] = -1e30
    criterion = nn.CrossEntropyLoss(reduction='mean')
    logits_aux = Variable(scores.data.new(scores.size(0), scores.size(1), 1).zero_())
    predictions = torch.cat([logits_aux, scores], dim=-1).contiguous()
    predictions = predictions.view(-1, 2)
    labels = labels.view(-1)
    loss = criterion.forward(predictions, labels)
    return loss

def adversarial_loss_computation(scores, labels, label_mask):
    scores = scores.suqeeze(dim=-1)
    logsigmoid_scores = F.logsigmoid(scores)
    positive_scores = logsigmoid_scores[labels==1]
    pos_loss = -positive_scores.mean()
    neg_label_mask = torch.logical_and(labels==0, label_mask==1)
    negative_scores = logsigmoid_scores[neg_label_mask]
    neg_loss = -negative_scores.mean()
    loss = (pos_loss + neg_loss)/ 2
    return loss

def rank_loss_computation(scores, labels, label_mask):
    scores = scores.squeeze(dim=-1)
    batch_size = scores.shape[0]
    positive_scores = scores[labels==1]
    print(positive_scores.shape)

    min_positive_scores = torch.min(positive_scores, dim=-1)[0]
    neg_label_mask = torch.logical_and(labels==0, label_mask==1)
    negative_scores = scores[neg_label_mask]
    print(negative_scores.shape)
    max_negative_scores = torch.max(negative_scores, dim=-1)[0]
    diff = max_negative_scores - min_positive_scores + 0.2
    print(diff.shape)
    loss = F.relu(diff).mean()
    return loss

def probe_model_evaluation(model, data_loader, args):
    model.eval()
    logs = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: batch[k].to(args.device) for k in batch}
            input = batch['input'].clamp(min=0)
            seq_mask = batch['seq_mask']
            attn_mask = (input >= 0)
            _, score = model(input=input, attn_mask=attn_mask, labels=batch['seq_labels'], label_mask=seq_mask)
            score[seq_mask == 0] = -1e30
            score = score.squeeze(-1)
            argsort = torch.argsort(score, dim=1, descending=True)
            batch_size = score.shape[0]
            for idx in range(batch_size):
                sorted_idx_i = argsort[idx]
                # input_i = input[idx]
                true_target_ids = input[idx][batch['seq_labels'][idx] == 1].tolist()
                # score_log = rank_contain_ratio_sorted_score(input=input_i, sorted_idx=sorted_idx_i, ground_truth_ids=true_target_ids)
                sorted_ids = input[idx][sorted_idx_i].tolist()
                score_log = rank_contain_ratio_score(pred_ids=sorted_ids, ground_truth_ids=true_target_ids, order=args.order)
                logs.append(score_log)
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    return metrics

# def em_f1_computation(pred_ids: list, true_ids: list):
#     em = em_score(prediction_tokens=pred_ids, groud_truth_tokens=true_ids)
#     f1, recall, prediction = f1_score(prediction_tokens=pred_ids, ground_truth_tokens=true_ids)
#     return em, f1

# def contain_ratio_score(pred_ids: list, ground_truth_ids: list):
#     topk = [3, 5, 10, 20, 50]
#     flag = False
#     idx = 0
#     contain_ratios = np.zeros(len(topk))
#     while (not flag) and (idx < len(topk)):
#         topk_ids = pred_ids[:topk[idx]]
#         flag = contains_subsequence(target=ground_truth_ids, sequence=topk_ids)
#         contain_ratios[idx] = float(flag)
#         idx = idx + 1
#     if flag:
#         for _ in range(idx, len(topk)):
#             contain_ratios[_] = 1
#     ratio_log = {}
#     for i, k in enumerate(topk):
#         ratio_log['Hit_{}'.format(k)] = contain_ratios[i]
#     return ratio_log

def rank_contain_ratio_score(pred_ids: list, ground_truth_ids: list, order=True):
    topk = [3, 5, 10, 20, 50]
    if order:
        contain_idx = find_subsequence(target=ground_truth_ids, sequence=pred_ids)
        contain_idx = contain_idx + 1
    else:
        contain_idx = find_subsequence_unorder(target=ground_truth_ids, sequence=pred_ids)
    ratio_log = {}
    for k in topk:
        if k >= contain_idx:
            ratio_log['Hit@{}'.format(k)] = 1.0
        else:
            ratio_log['Hit@{}'.format(k)] = 0.0
    if contain_idx > len(pred_ids):
        ratio_log['MRR'] = 0.0
    else:
        ratio_log['MRR'] = 1.0/contain_idx
    ratio_log['MR'] = contain_idx
    return ratio_log



# def rank_contain_ratio_sorted_score(input: Tensor, sorted_idx: Tensor, ground_truth_ids: list):
#     topk = [3, 5, 10, 20, 50]
#     seq_len = input.shape[0]
#     contain_idx = 300
#     for rank in topk:
#         sorted_idx = sorted_idx[:rank]
#         # zero_seq = torch.zeros(seq_len, dtype=torch.long)
#         # zero_seq[sorted_idx] = 1
#         # inp_seq = input[zero_seq==1].tolist()
#         sorted_sorted_idx = torch.sort(sorted_idx)[0]
#         inp_seq = input[sorted_sorted_idx].tolist()
#         if contains_subsequence(target=ground_truth_ids, sequence=inp_seq):
#             contain_idx = rank
#             break
#     ratio_log = {}
#     for k in topk:
#         if k >= contain_idx:
#             ratio_log['Hit@{}'.format(k)] = 1.0
#         else:
#             ratio_log['Hit@{}'.format(k)] = 0.0
#     if contain_idx > seq_len:
#         ratio_log['MRR'] = 0.0
#     else:
#         ratio_log['MRR'] = 1.0/contain_idx
#     ratio_log['MR'] = contain_idx
#     return ratio_log

# def em_score(prediction_tokens, groud_truth_tokens):
#     if len(prediction_tokens) != len(groud_truth_tokens):
#         return 0.0
#     for idx in range(len(prediction_tokens)):
#         if prediction_tokens[idx] != groud_truth_tokens[idx]:
#             return 0.0
#     return 1.0


# def f1_score(prediction_tokens, ground_truth_tokens):
#     ZERO_METRIC = (0, 0, 0)
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return ZERO_METRIC
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1, precision, recall

if __name__ == '__main__':
    target = [1,3,4]
    sequence = [0, 0, 0, 1, 4, 0, 1, 2,  5, 3, 4, 4]
    x = find_subsequence(target=target, sequence=sequence)
    print(x)