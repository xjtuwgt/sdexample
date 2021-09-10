import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import transformers
import random
import numpy as np
from os.path import join
from envs import HOME_DATA_FOLDER, OUTPUT_FOLDER
from utils.env_utils import seed_everything

from data_utils.findcat import FindCatDataset, find_cat_validation_fn, find_cat_collate_fn
from data_utils.dataset import SentenceDropDataset

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
                             multi_target=args.multi_target in ['multi'],
                             data_file_name=train_file_name)
    validation_fn = find_cat_validation_fn if args.validate_examples else lambda ex: True
    sdrop_dataset = SentenceDropDataset(dataset=dataset,
                                        sent_drop_prob=args.sent_dropout,
                                        beta_drop=args.beta_drop,
                                        mask=args.mask,
                                        mask_id=args.mask_id,
                                        example_validate_fn=validation_fn)
    dataloader = DataLoader(sdrop_dataset,
                            shuffle=True,
                            batch_size=args.batch_size,
                            collate_fn=find_cat_collate_fn)
    return dataloader

def dev_data_loader(args):
    dev_seq_len = args.eval_test_seq_len
    dev_seq_len = [int(_) for _ in dev_seq_len.split(',')]
    if args.test_file_name is not None:
        dev_file_name = join(HOME_DATA_FOLDER, 'toy_data', args.eval_file_name)
        print('Dev data file name = {}'.format(dev_file_name))
    else:
        dev_file_name = None
    dataset = FindCatDataset(seed=2345,
                             seqlen=dev_seq_len,
                             total_examples=args.test_examples,
                             multi_target=args.multi_target in ['multi'],
                             data_file_name=dev_file_name)
    dev_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, collate_fn=find_cat_collate_fn)
    return dev_dataloader

def test_data_loader(args):
    test_seq_len = args.eval_test_seq_len
    test_seq_len = [int(_) for _ in test_seq_len.split(',')]
    if args.test_file_name is not None:
        test_file_name = join(HOME_DATA_FOLDER, 'toy_data', args.test_file_name)
        print('test data file name = {}'.format(test_file_name))
    else:
        test_file_name = None
    dataset = FindCatDataset(seed=1234,
                             seqlen=test_seq_len,
                             total_examples=args.test_examples,
                             multi_target=args.multi_target in ['multi'],
                             data_file_name=test_file_name)
    test_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, collate_fn=find_cat_collate_fn)
    return test_dataloader

def save_match_model(model, model_name):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
        torch.save({k: v.cpu() for k, v in model.module.state_dict().items()}, model_name)
    else:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_name)
    print('Saving model at {}'.format(model_name))

def model_evaluation(model, data_loader, args):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: batch[k].to(args.device) for k in batch}
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            _, logits = model(input, attention_mask=attn_mask, labels=batch['labels'])
            pred = logits.max(1)[1]
            total += len(pred)
            correct += (pred == batch['labels']).sum()
    dev_acc = correct * 1.0 / total
    return dev_acc
