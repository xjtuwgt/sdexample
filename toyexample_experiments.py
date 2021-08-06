from argparse import ArgumentParser
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import transformers
import random
import numpy as np
from os.path import join
from envs import HOME_DATA_FOLDER
from utils.gpu_utils import get_single_free_gpu

from data_utils.findcat import FindCatDataset, find_cat_validation_fn, find_cat_collate_fn
from data_utils.dataset import SentenceDropDataset

def seed_everything(seed: int) -> int:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

def model_builder(args):
    model_config = transformers.AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = 3
    model_config.vocab_size = args.vocab_size
    model = transformers.AutoModelForSequenceClassification.from_config(model_config)
    print('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)
    return model

def train_data_loader(args):
    train_seq_len = args.train_seq_len
    train_seq_len = [int(_) for _ in train_seq_len.split(',')]
    if args.train_file_name is not None:
        train_file_name = join(HOME_DATA_FOLDER, 'toy_data', args.train_file_name)
    else:
        train_file_name = None
    dataset = FindCatDataset(seed=args.seed,
                             target_tokens=args.target_tokens,
                             seqlen=train_seq_len,
                             total_examples=args.train_examples,
                             data_file_name=train_file_name)
    validation_fn = find_cat_validation_fn if args.validate_examples else lambda ex: True
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=args.sent_dropout,
        example_validate_fn=validation_fn)
    dataloader = DataLoader(sdrop_dataset, batch_size=args.batch_size, collate_fn=find_cat_collate_fn)
    return dataloader

def dev_data_loader(args):
    dev_seq_len = args.train_seq_len
    dev_seq_len = [int(_) for _ in dev_seq_len.split(',')]
    if args.test_file_name is not None:
        dev_file_name = join(HOME_DATA_FOLDER, 'toy_data', args.test_file_name)
    else:
        dev_file_name = None
    dataset = FindCatDataset(seed=1234,
                             seqlen=dev_seq_len,
                             total_examples=args.test_examples,
                             data_file_name=dev_file_name)
    dev_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, collate_fn=find_cat_collate_fn)
    return dev_dataloader

def complete_default_parser(args):
    idx, used_memory = get_single_free_gpu()
    device = torch.device("cuda:{}".format(idx) if torch.cuda.is_available() else "cpu")
    args.device = device
    return args

if __name__ == "__main__":
    parser = ArgumentParser()
    ##train data set
    parser.add_argument('--target_tokens', type=str, default='cat')
    parser.add_argument('--sent_dropout', type=float, default=0.1)
    parser.add_argument('--train_examples', type=int, default=100)
    parser.add_argument('--train_seq_len', type=str, default='300')
    parser.add_argument('--train_file_name', type=str, default='train_cat_100_42_300_0.5.pkl.gz')

    ##test data set
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--vocab_size', type=int, default=100) ## 100
    parser.add_argument('--test_seq_len', type=str, default='300')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=300)
    parser.add_argument('--test_file_name', type=str, default='test_cat_10000_1234_300_0.5.pkl.gz')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_interval_num', type=int, default=100)

    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    args = complete_default_parser(args=args)
    train_dataloader = train_data_loader(args=args)
    dev_dataloader = dev_data_loader(args=args)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # torch.cuda.manual_seed_all(args.seed)
    seed_everything(seed=args.seed)
    model = model_builder(args=args)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    step = 0
    start_epoch = 0
    best_dev_acc = -1
    best_step = None
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_iterator = trange(start_epoch, start_epoch + int(args.epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        train_correct = 0
        train_total = 0
        for batch_idx, batch in enumerate(epoch_iterator):
            model.train()
            batch = {k: batch[k].to(args.device) for k in batch}
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            loss, logits = model(input, attention_mask=attn_mask, labels=batch['labels'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pred = logits.max(1)[1]
            train_total = train_total + pred.shape[0]
            train_correct += (pred == batch['labels']).sum()

            if (step + 1) % args.eval_batch_interval_num == 0:
                model.eval()
                total = 0
                correct = 0
                with torch.no_grad():
                    for batch in tqdm(dev_dataloader):
                        batch = {k: batch[k].to(args.device) for k in batch}
                        input = batch['input'].clamp(min=0)
                        attn_mask = (input >= 0)
                        _, logits = model(input, attention_mask=attn_mask, labels=batch['labels'])
                        pred = logits.max(1)[1]
                        total += len(pred)
                        correct += (pred == batch['labels']).sum()
                print("Step {}: dev accuracy={:.6f}".format((epoch, step), correct*1.0/total), flush=True)
                dev_acc = correct*1.0 / total
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_step = (epoch, step)
            step = step + 1
        print('Train accuracy = {:.6f} at {}'.format(train_correct *1.0 /train_total, epoch))
    print("Best dev result at {} dev accuracy={:.6f} at step".format(best_step, best_dev_acc))