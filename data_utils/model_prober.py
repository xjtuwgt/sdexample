from torch import nn
import argparse
import torch
from data_utils.model_utils import model_builder, load_pretrained_model
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification

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
    return parser

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
            self.encoder.load_state_dict(torch.load(self.config.pre_trained_file_name))
        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_labels)


    def forward(self, input, attn_mask):
        self.model(input, attn_mask)
        bert_output = activation['bert']
        seq_output = bert_output[0]
        seq_output = self.dropout(seq_output)
        seq_scores = self.classifier(seq_output).squeeze(dim=-1)
        return seq_scores