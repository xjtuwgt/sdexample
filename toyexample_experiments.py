from argparse import ArgumentParser
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
import random
import numpy as np
from os.path import join
from envs import HOME_DATA_FOLDER

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
    # if args.train_file_name is not None:
    #     train_file_name =
    dataset = FindCatDataset(seed=args.seed,
                             target_tokens=args.target_tokens,
                             seqlen=train_seq_len,
                             total_examples=args.train_examples)
    validation_fn = find_cat_validation_fn if args.validate_examples else lambda ex: True
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=args.sent_dropout,
        example_validate_fn=validation_fn)
    dataloader = DataLoader(sdrop_dataset, batch_size=args.batch_size, collate_fn=find_cat_collate_fn)
    return dataloader

def dev_data_loader(args):
    dataset = FindCatDataset(seed=314,
                   seqlen=args.seq_len,
                   total_examples=args.test_examples)
    dev_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, collate_fn=find_cat_collate_fn)
    return dev_dataloader

def complete_default_parser(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # set n_gpu
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.data_parallel:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    return args

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu_id", default=None, type=str, help="GPU id")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--data_parallel", type=bool, default=False)
    ##train data set
    parser.add_argument('--target_tokens', type=str, default='cat')
    parser.add_argument('--sent_dropout', type=float, default=.1)
    parser.add_argument('--train_examples', type=int, default=1000)
    parser.add_argument('--train_seq_len', type=str, default='300')
    parser.add_argument('--train_file_name', type=str, default='test_cat_10000_1234_300_0.5.pkl.gz')

    ##test data set
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=100) ## 100
    parser.add_argument('--test_seq_len', type=str, default='300')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=300)
    parser.add_argument('--test_file_name', type=str, default='test_cat_10000_1234_300_0.5.pkl.gz')

    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    args = complete_default_parser(args=args)
    dataloader = train_data_loader(args=args)
    dev_dataloader = dev_data_loader(args=args)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # torch.cuda.manual_seed_all(args.seed)
    seed_everything(seed=args.seed)
    model = model_builder(args=args)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    step = 0
    best_dev_acc = -1
    best_step = -1

    total = 0
    correct = 0
    total_loss = 0

    while True:
        model.train()
        for batch in tqdm(dataloader):
            batch = {k: batch[k].to(args.device) for k in batch}
            step += 1
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            loss, logits = model(input, attention_mask=attn_mask, labels=batch['labels'])
            # print(output)

            # total_loss += output[0].loss.item()
            total_loss += loss.item()
            print(f"Step {step:6d}: loss={loss.item()}")
            optimizer.zero_grad()
            # output.loss.backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred = logits.max(1)[1]
            total += len(pred)
            correct += (pred == batch['labels']).sum()

            if step % args.eval_every == 0:
                print(f"Step {step}: train accuracy={correct / total:.6f}, train loss={total_loss / total}", flush=True)
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

                print(f"Step {step}: dev accuracy={correct / total:.6f}", flush=True)

                if correct / total > best_dev_acc:
                    best_dev_acc = correct / total
                    best_step = step

                total = 0
                correct = 0
                total_loss = 0

            if step >= args.steps:
                break

        if step >= args.steps:
            break

    print(f"Best dev result: dev accuracy={best_dev_acc:.6f} at step {best_step}")