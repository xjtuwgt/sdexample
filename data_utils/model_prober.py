from torch import nn
import argparse
import torch
from data_utils.model_utils import model_builder, load_pretrained_model
from utils.gpu_utils import get_single_free_gpu
from torch.autograd import Variable
from tqdm import tqdm
from envs import OUTPUT_FOLDER
from os.path import join


def prober_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--train_file_name', type=str, default='train_fastsingle_cat_100_42_300_0.5.pkl.gz')
    # parser.add_argument('--train_file_name', type=str, default='train_single_ant_bear_cat_dog_eagle_fox_goat_horse_indri_jaguar_koala_lion_moose_numbat_otter_pig_quail_rabbit_shark_tiger_uguisu_wolf_xerus_yak_zebra_30000_42_300_0.5.pkl.gz')
    parser.add_argument('--vocab_size', type=int, default=100) ## 100
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=300)
    parser.add_argument('--eval_file_name', type=str, default='eval_fastsingle_cat_10000_2345_300_0.5.pkl.gz')
    parser.add_argument('--test_file_name', type=str, default='test_fastsingle_cat_10000_1234_300_0.5.pkl.gz')
    parser.add_argument('--pre_trained_file_name', type=str, default=None)
    parser.add_argument('--dropout_prob', type=float, default=0.25)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--topk', type=int, default=len('cat'))
    return parser

def complete_default_parser(args):
    if torch.cuda.is_available():
        idx, used_memory = get_single_free_gpu()
        # idx = 0
        device = torch.device("cuda:{}".format(idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    args.device = device
    return args

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
        self.model.bert.register_forward_hook(get_activation('bert'))
        if self.config.pre_trained_file_name is not None:
            self.encoder.load_state_dict(torch.load(join(OUTPUT_FOLDER, self.config.pre_trained_file_name)))
        for param in self.model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_labels)

    def forward(self, input, attn_mask, labels):
        self.model(input, attn_mask)
        bert_output = activation['bert']
        seq_output = bert_output[0]
        seq_output = self.dropout(seq_output)
        seq_scores = self.classifier(seq_output)
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
    em = []
    f1 = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: batch[k].to(args.device) for k in batch}
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            _, logits = model(input, attention_mask=attn_mask, labels=batch['seq_labels'])
            _, pred_topk_idxes = torch.topk(input=logits, k=args.topk)
            batch_size = logits.shape[0]
            for idx in range(batch_size):
                pred_topk_i = pred_topk_idxes[idx]
                true_label_i = batch['seq_labels'][idx]
                inter_count = true_label_i[pred_topk_i].sum().item()
                if inter_count == args.topk:
                    em.append(1.0)
                else:
                    em.append(0.0)
                f1.append(inter_count * 1.0 / args.topk)
    assert len(em) == len(f1)
    em = sum(em)/len(em)
    f1 = sum(f1)/len(f1)
    return em, f1