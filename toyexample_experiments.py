from argparse import ArgumentParser
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import transformers
import random
import numpy as np
from os.path import join
from envs import HOME_DATA_FOLDER, OUTPUT_FOLDER
from utils.gpu_utils import get_single_free_gpu
from data_utils.findcat import MASK

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

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

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

def complete_default_parser(args):
    if torch.cuda.is_available():
        idx, used_memory = get_single_free_gpu()
        # idx = 0
        device = torch.device("cuda:{}".format(idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    args.device = device
    return args

def save_match_model(model, model_name):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
        torch.save({k: v.cpu() for k, v in model.module.model.state_dict().items()}, model_name)
    else:
        torch.save({k: v.cpu() for k, v in model.model.state_dict().items()}, model_name)
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


def default_argparser():
    parser = ArgumentParser()
    ##train data set
    parser.add_argument('--target_tokens', type=str, default='cat')
    parser.add_argument('--sent_dropout', type=float, default=0.0)
    parser.add_argument('--beta_drop', type=boolean_string, default='false')
    parser.add_argument('--mask', type=boolean_string, default='false')
    parser.add_argument('--mask_id', type=int, default=MASK)
    parser.add_argument('--train_examples', type=int, default=500)
    parser.add_argument('--multi_target', type=str, default='multi')
    parser.add_argument('--train_seq_len', type=str, default='300')
    parser.add_argument('--train_file_name', type=str, default='train_fastsingle_cat_500_42_300_0.5.pkl.gz')
    # parser.add_argument('--train_file_name', type=str, default='train_single_ant_bear_cat_dog_eagle_fox_goat_horse_indri_jaguar_koala_lion_moose_numbat_otter_pig_quail_rabbit_shark_tiger_uguisu_wolf_xerus_yak_zebra_30000_42_300_0.5.pkl.gz')

    ##test data set
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--vocab_size', type=int, default=100) ## 100
    parser.add_argument('--eval_test_seq_len', type=str, default='300')
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=300)
    parser.add_argument('--eval_file_name', type=str, default='eval_fastsingle_cat_10000_2345_300_0.5.pkl.gz')
    parser.add_argument('--test_file_name', type=str, default='test_fastsingle_cat_10000_1234_300_0.5.pkl.gz')

    # parser.add_argument('--eval_file_name', type=str, default='eval_single_snake_robin_puma_oyster_ibis_10000_2345_300_0.5.pkl.gz')
    # parser.add_argument('--test_file_name', type=str, default='test_single_snake_robin_puma_oyster_ibis_10000_1234_300_0.5.pkl.gz')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=10000000000)
    parser.add_argument('--eval_batch_interval_num', type=int, default=100)

    parser.add_argument("--data_parallel",
                        default='false',
                        type=boolean_string,
                        help="use data parallel or not")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_model', type=bool, default=False)

    return parser