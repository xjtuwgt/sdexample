from data_utils.argument_parser import data_aug_default_parser, complete_default_parser
from toyexample_experiments import seed_everything, dev_data_loader
from data_utils.da_metrics_utils import affinity_metrics_computation, diversity_metrics_computation
import torch
from data_utils.model_utils import load_pretrained_model
from envs import OUTPUT_FOLDER
from tqdm import tqdm, trange
from os.path import join
import os
from data_utils.da_metrics_utils import MODEL_NAMES
model_dict = {_[0]: _[1] for _ in MODEL_NAMES}

def accuracy_collection(args):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # if args.exp_name is None:
    args.exp_name = args.train_file_name + '.models'
    model_name_dict = model_dict[args.exp_name]
    args.exp_name = join(OUTPUT_FOLDER, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    orig_model_name = join(args.exp_name, model_name_dict['orig'])
    beta_drop_model_name = join(args.exp_name, model_name_dict['beta_drop'])
    drop_model_name = join(args.exp_name, model_name_dict['drop'])
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 50)
    for idx, (key, value) in enumerate(model_dict.items()):
        for k, v in value.items():
            print(idx + 1, key, k, v)
    print('*' * 50)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dev_dataloader = dev_data_loader(args=args)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    seed_everything(seed=args.seed)
    orig_model = load_pretrained_model(args=args, pretrained_model_name=orig_model_name)
    drop_model = load_pretrained_model(args=args, pretrained_model_name=drop_model_name)
    beta_drop_model = load_pretrained_model(args=args, pretrained_model_name=beta_drop_model_name)



if __name__ == '__main__':
    train_file_names = ['train_fastsingle_cat_100_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_200_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_500_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_1000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_2000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_5000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_10000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_20000_42_300_0.5.pkl.gz']

    dev_file_names = []
    parser = data_aug_default_parser()
    args = parser.parse_args()
    args = complete_default_parser(args=args)