from torch import nn
import argparse
import torch
from data_utils.model_utils import model_builder, load_pretrained_model
from utils.gpu_utils import get_single_free_gpu
from torch.autograd import Variable
from tqdm import tqdm
from envs import OUTPUT_FOLDER
from os.path import join
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

ORIG_PROBE_MODEL_NAME = join(OUTPUT_FOLDER, 'model_False_0.0_384_221_dev_0.9792.pkl')
DROP_PROBE_MODEL_NAME = join(OUTPUT_FOLDER, 'model_True_0.1_97_52_dev_0.9901.pkl')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

class OutputLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.25, num_answer=1):
        super(OutputLayer, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim*2, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, num_answer),
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)

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
        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_labels)
        # self.classifier = OutputLayer(hidden_dim=self.config.hidden_dim, dropout=self.config.dropout_prob, num_answer=self.config.num_labels)

    def forward(self, input, attn_mask, labels, label_mask):
        self.model(input, attn_mask)
        # bert_output = activation['bert']
        bert_output = activation['encoder']
        # print(len(bert_output))
        # print(bert_output[0].shape)
        seq_output = bert_output[0]
        seq_output = self.dropout(seq_output)
        seq_scores = self.classifier(seq_output)
        seq_scores[label_mask==0] = -1e30
        loss = loss_computation(scores=seq_scores, labels=labels)
        return loss, seq_scores

def loss_computation(scores, labels):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    logits_aux = Variable(scores.data.new(scores.size(0), scores.size(1), 1).zero_())
    predictions = torch.cat([logits_aux, scores], dim=-1).contiguous()
    predictions = predictions.view(-1, 2)
    labels = labels.view(-1)
    loss = criterion.forward(predictions, labels)
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
            score = score.squeeze(-1)
            argsort = torch.argsort(score, dim=1, descending=True)
            batch_size = score.shape[0]
            for idx in range(batch_size):
                sorted_idx_i = argsort[idx]
                input_i = input[idx]
                true_target_ids = input[idx][batch['seq_labels'][idx] == 1].tolist()
                score_log = rank_contain_ratio_sorted_score(input=input_i, sorted_idx=sorted_idx_i, ground_truth_ids=true_target_ids)
                # sorted_ids = input[idx][sorted_idx_i].tolist()
                # true_target_ids = input[idx][batch['seq_labels'][idx] == 1].tolist()
                # score_log = rank_contain_ratio_score(pred_ids=sorted_ids, ground_truth_ids=true_target_ids)
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

# def rank_contain_ratio_score(pred_ids: list, ground_truth_ids: list):
#     topk = [3, 5, 10, 20, 50]
#     contain_idx = find_subsequence(target=ground_truth_ids, sequence=pred_ids)
#     contain_idx = contain_idx + 1
#     ratio_log = {}
#     for k in topk:
#         if k >= contain_idx:
#             ratio_log['Hit@{}'.format(k)] = 1.0
#         else:
#             ratio_log['Hit@{}'.format(k)] = 0.0
#     if contain_idx > len(pred_ids):
#         ratio_log['MRR'] = 0.0
#     else:
#         ratio_log['MRR'] = 1.0/contain_idx
#     ratio_log['MR'] = contain_idx
#     return ratio_log

def rank_contain_ratio_sorted_score(input: Tensor, sorted_idx: Tensor, ground_truth_ids: list):
    topk = [3, 5, 10, 20, 50]
    seq_len = input.shape[0]
    contain_idx = seq_len
    for rank in topk:
        sorted_idx = sorted_idx[:rank]
        zero_seq = torch.zeros(seq_len, dtype=torch.long)
        zero_seq[sorted_idx] = 1
        inp_seq = input[zero_seq==1].tolist()
        # sorted_sorted_idx = torch.sort(sorted_idx)[0]
        # inp_seq = input[sorted_sorted_idx].tolist()
        if contains_subsequence(target=ground_truth_ids, sequence=inp_seq):
            contain_idx = rank
            break
    ratio_log = {}
    for k in topk:
        if k >= contain_idx:
            ratio_log['Hit@{}'.format(k)] = 1.0
        else:
            ratio_log['Hit@{}'.format(k)] = 0.0
    if contain_idx > seq_len:
        ratio_log['MRR'] = 0.0
    else:
        ratio_log['MRR'] = 1.0/contain_idx
    ratio_log['MR'] = contain_idx
    return ratio_log

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