from argparse import ArgumentParser
from data_utils.findcat import MASK
import torch
from utils.gpu_utils import get_single_free_gpu
from utils.env_utils import seed_everything

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def complete_default_parser(args):
    seed_everything(seed=args.seed)
    if torch.cuda.is_available():
        idx, used_memory = get_single_free_gpu()
        # idx = 0
        device = torch.device("cuda:{}".format(idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    args.device = device
    return args

def prober_default_parser():
    parser = ArgumentParser()
    parser.add_argument('--train_examples', type=int, default=800)
    parser.add_argument('--multi_target', type=str, default='multi')
    parser.add_argument('--train_seq_len', type=str, default='300')
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--eval_test_seq_len', type=str, default='300')
    parser.add_argument('--target_tokens', type=str, default='cat')

    parser.add_argument('--lstm_hidden_dim', type=int, default=300)
    parser.add_argument('--lstm_layers', type=int, default=2)


    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--train_file_name', type=str, default='train_fastsingle_cat_5000_42_300_1.0.pkl.gz')
    parser.add_argument('--vocab_size', type=int, default=100) ## 100
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=300)
    parser.add_argument('--eval_file_name', type=str, default='eval_fastsingle_cat_10000_2345_300_1.0.pkl.gz')
    parser.add_argument('--test_file_name', type=str, default='test_fastsingle_cat_10000_1234_300_1.0.pkl.gz')
    parser.add_argument('--pre_trained_file_name', type=str, default=None)
    parser.add_argument('--drop_model', type=boolean_string, default='true')
    parser.add_argument('--order', type=boolean_string, default='true')
    parser.add_argument('--loss_type', type=str, default='bce')
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--topk', type=int, default=len('cat'))
    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--adversarial', type=boolean_string, default='true')
    parser.add_argument('--adversarial_temp', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_interval_num', type=int, default=300)
    parser.add_argument('--window_size', type=int, default=10000000000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1234)
    return parser

def data_aug_default_parser():
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_examples', type=int, default=800)
    parser.add_argument('--multi_target', type=str, default='multi')
    parser.add_argument('--train_seq_len', type=str, default='300')
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--eval_test_seq_len', type=str, default='300')
    parser.add_argument('--target_tokens', type=str, default='cat')

    parser.add_argument('--sent_dropout', type=float, default=0.1)
    parser.add_argument('--beta_drop', type=boolean_string, default='false')
    parser.add_argument('--mask', type=boolean_string, default='false')
    parser.add_argument('--mask_id', type=int, default=MASK)

    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--orig_model_name', type=str, default=None)
    parser.add_argument('--drop_model_name', type=str, default=None)
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--train_file_name', type=str, default='train_fastsingle_cat_100_42_300_0.5.pkl.gz')
    # parser.add_argument('--train_file_name', type=str, default='train_single_ant_bear_cat_dog_eagle_fox_goat_horse_indri_jaguar_koala_lion_moose_numbat_otter_pig_quail_rabbit_shark_tiger_uguisu_wolf_xerus_yak_zebra_30000_42_300_0.5.pkl.gz')
    parser.add_argument('--vocab_size', type=int, default=100) ## 100
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_every', type=int, default=300)
    parser.add_argument('--eval_file_name', type=str, default='eval_fastsingle_cat_10000_2345_300_0.5.pkl.gz')
    parser.add_argument('--test_file_name', type=str, default='test_fastsingle_cat_10000_1234_300_0.5.pkl.gz')
    parser.add_argument('--pre_trained_file_name', type=str, default=None)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--topk', type=int, default=len('cat'))

    parser.add_argument('--seed', type=int, default=1234)
    return parser


def default_argparser():
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=None)
    ##train data set
    parser.add_argument('--target_tokens', type=str, default='cat')
    parser.add_argument('--sent_dropout', type=float, default=0.0)
    parser.add_argument('--beta_drop', type=boolean_string, default='false')
    parser.add_argument('--mask', type=boolean_string, default='false')
    parser.add_argument('--mask_id', type=int, default=MASK)
    parser.add_argument('--train_examples', type=int, default=800)
    parser.add_argument('--multi_target', type=str, default='multi')
    parser.add_argument('--train_seq_len', type=str, default='300')
    parser.add_argument('--train_file_name', type=str, default='train_fastsingle_cat_500_42_300_0.5.pkl.gz')
    # parser.add_argument('--train_file_name', type=str, default='train_single_ant_bear_cat_dog_eagle_fox_goat_horse_indri_jaguar_koala_lion_moose_numbat_otter_pig_quail_rabbit_shark_tiger_uguisu_wolf_xerus_yak_zebra_40000_42_300_0.5.pkl.gz')

    ##test data set
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--vocab_size', type=int, default=100) ## 100
    parser.add_argument('--eval_test_seq_len', type=str, default='300')
    parser.add_argument('--epochs', type=int, default=300)
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
    parser.add_argument('--save_model', type=boolean_string, default='true')

    return parser