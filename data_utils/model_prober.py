from torch import nn
import argparse
import torch
from data_utils.model_utils import model_builder, load_pretrained_model
from utils.gpu_utils import get_single_free_gpu
from torch.autograd import Variable
from tqdm import tqdm
from envs import OUTPUT_FOLDER
from os.path import join
from collections import Counter

ORIG_PROBE_MODEL_NAME = join(OUTPUT_FOLDER, 'model_False_0.0_384_221_dev_0.9792.pkl')
DROP_PROBE_MODEL_NAME = join(OUTPUT_FOLDER, 'model_True_0.1_97_52_dev_0.9901.pkl')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

class ProberModel(nn.Module):
    def __init__(self, config):
        super(ProberModel, self).__init__()
        self.config = config
        self.model = model_builder(args=self.config)
        if self.config.pre_trained_file_name is not None:
            self.model.load_state_dict(torch.load(self.config.pre_trained_file_name))
            print('Loading model from {}'.format(self.config.pre_trained_file_name))
        self.model.bert.register_forward_hook(get_activation('bert'))
        for param in self.model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_labels)

    def forward(self, input, attn_mask, labels, label_mask):
        self.model(input, attn_mask)
        bert_output = activation['bert']
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
    em_list = []
    f1_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: batch[k].to(args.device) for k in batch}
            input = batch['input'].clamp(min=0)
            seq_mask = batch['seq_mask']
            attn_mask = (input >= 0)
            _, logits = model(input=input, attn_mask=attn_mask, labels=batch['seq_labels'], label_mask=seq_mask)
            sig_logits = torch.sigmoid(logits)
            _, pred_topk_idxes = torch.topk(input=sig_logits, k=args.topk)
            batch_size = logits.shape[0]
            for idx in range(batch_size):
                pred_topk_i = pred_topk_idxes[idx]
                pred_ids = input[idx][pred_topk_i].tolist()
                true_target_ids = input[idx][batch['seq_labels'][idx]==1].tolist()
                em, f1 = em_f1_computation(pred_ids=pred_ids, true_ids=true_target_ids)
                em_list.append(em)
                f1_list.append(f1)
    assert len(em_list) == len(f1_list)
    em_ = sum(em_list)/len(em_list)
    f1_ = sum(f1_list)/len(f1_list)
    return em_, f1_

def em_f1_computation(pred_ids: list, true_ids: list):
    em = em_score(prediction_tokens=pred_ids, groud_truth_tokens=true_ids)
    f1, recall, prediction = f1_score(prediction_tokens=pred_ids, ground_truth_tokens=true_ids)
    return em, f1

def em_score(prediction_tokens, groud_truth_tokens):
    if len(prediction_tokens) != len(groud_truth_tokens):
        return 0.0
    for idx in range(len(prediction_tokens)):
        if prediction_tokens[idx] != groud_truth_tokens[idx]:
            return 0.0
    return 1.0

def f1_score(prediction_tokens, ground_truth_tokens):
    ZERO_METRIC = (0, 0, 0)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall