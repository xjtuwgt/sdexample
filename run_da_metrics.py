from data_utils.argument_parser import data_aug_default_parser, complete_default_parser
from data_utils.da_metrics_utils import orig_da_train_data_loader, orig_da_dev_data_loader, drop_da_train_data_loader, drop_da_dev_data_loader
from toyexample_experiments import seed_everything
from data_utils.da_metrics_utils import affinity_metrics_computation, diversity_metrics_computation
import torch
from data_utils.model_utils import load_pretrained_model
from envs import OUTPUT_FOLDER
from tqdm import tqdm, trange
from os.path import join
import os
from data_utils.da_metrics_utils import MODEL_NAMES

model_dict = {_[0]: _[1] for _ in MODEL_NAMES}

def da_affinity_metrics_collection(args):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # if args.exp_name is None:
    args.exp_name = args.train_file_name + '.models'
    model_name_dict = model_dict[args.exp_name]
    args.exp_name = join(OUTPUT_FOLDER, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    args.orig_model_name = join(args.exp_name, model_name_dict['orig'])
    if args.beta_drop:
        args.drop_model_name = join(args.exp_name, model_name_dict['beta_drop'])
    else:
        args.drop_model_name = join(args.exp_name, model_name_dict['drop'])
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 50)
    for idx, (key, value) in enumerate(model_dict.items()):
        for k, v in value.items():
            print(idx + 1, key, k, v)
    print('*' * 50)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # orig_train_dataloader = orig_da_train_data_loader(args=args)
    orig_dev_dataloader = orig_da_dev_data_loader(args=args)

    # drop_train_dataloader = drop_da_train_data_loader(args=args)
    drop_dev_data_loader = drop_da_dev_data_loader(args=args)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    seed_everything(seed=args.seed)
    if args.orig_model_name is not None:
        orig_model = load_pretrained_model(args=args, pretrained_model_name=args.orig_model_name)
        affinity_metrics = affinity_metrics_computation(model=orig_model, dev_data_loader=orig_dev_dataloader,
                                                        drop_dev_data_loader=drop_dev_data_loader, args=args)
    else:
        affinity_metrics = 0.0

    return affinity_metrics

def da_diversity_metrics_collection(args):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # if args.exp_name is None:
    args.exp_name = args.train_file_name + '.models'
    model_name_dict = model_dict[args.exp_name]
    args.exp_name = join(OUTPUT_FOLDER, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    args.orig_model_name = join(args.exp_name, model_name_dict['orig'])
    if args.beta_drop:
        args.drop_model_name = join(args.exp_name, model_name_dict['beta_drop'])
    else:
        args.drop_model_name = join(args.exp_name, model_name_dict['drop'])
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 50)
    for idx, (key, value) in enumerate(model_dict.items()):
        for k, v in value.items():
            print(idx + 1, key, k, v)
    print('*' * 50)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    orig_train_dataloader = orig_da_train_data_loader(args=args)
    drop_train_dataloader = drop_da_train_data_loader(args=args)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    seed_everything(seed=args.seed)
    if args.drop_model_name is not None and args.orig_model_name is not None:
        drop_model = load_pretrained_model(args=args, pretrained_model_name=args.drop_model_name)
        orig_model = load_pretrained_model(args=args, pretrained_model_name=args.orig_model_name)
        diversity_metrics = diversity_metrics_computation(model=orig_model, drop_model=drop_model,
                                                          train_data_loader=orig_train_dataloader,
                                                          drop_train_data_loader=drop_train_dataloader,
                                                          args=args)
    else:
        diversity_metrics = 0.0
    return diversity_metrics


if __name__ == '__main__':
    train_file_names = ['train_fastsingle_cat_100_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_200_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_500_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_1000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_2000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_5000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_10000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_20000_42_300_0.5.pkl.gz']
    # train_file_names = ['train_fastsingle_cat_1000_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_1000_43_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_1000_44_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_1000_45_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_1000_4_300_0.5.pkl.gz',]
    parser = data_aug_default_parser()
    args = parser.parse_args()
    args = complete_default_parser(args=args)
    # da_metric_list = []
    # for train_name in train_file_names:
    #     args.train_file_name = train_name
    #     aff_metric = da_affinity_metrics_collection(args=args)
    #     da_metric_list.append((train_name, aff_metric))
    #     print('+' * 100)
    # for name, metric in da_metric_list:
    #     print(name, metric)

    di_metric_list = []
    for train_name in train_file_names:
        args.train_file_name = train_name
        div_metric = da_diversity_metrics_collection(args=args)
        di_metric_list.append((train_name, div_metric))
        print('+' * 100)
    for name, metric in di_metric_list:
        print(name, metric)

    # if args.drop_model_name is not None and args.orig_model_name is not None:
    #     drop_model = load_pretrained_model(args=args, pretrained_model_name=args.drop_model_name)
    #     orig_model = load_pretrained_model(args=args, pretrained_model_name=args.orig_model_name)
    #     diversity_metrics = diversity_metrics_computation(model=orig_model, drop_model=drop_model,
    #                                                       train_data_loader=orig_train_dataloader,
    #                                                       drop_train_data_loader=drop_train_dataloader,
    #                                                       args=args)
    # else:
    #     diversity_metrics = 0.0

    # for key, value in vars(args).items():
    #     print(key, value)
    # print('Affinity metrics: {}'.format(affinity_metrics))
    # print('Diversity metrics: {}'.format(diversity_metrics))


# parser = data_aug_default_parser()
# args = parser.parse_args()
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# args = complete_default_parser(args=args)
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# if args.exp_name is None:
#     args.exp_name = args.train_file_name + '.models'
#     model_name_dict = model_dict[args.exp_name]
#     args.exp_name = join(OUTPUT_FOLDER, args.exp_name)
#     os.makedirs(args.exp_name, exist_ok=True)
#     args.orig_model_name = join(args.exp_name, model_name_dict['orig'])
#     if args.beta_drop:
#         args.drop_model_name = join(args.exp_name, model_name_dict['beta_drop'])
#     else:
#         args.drop_model_name = join(args.exp_name, model_name_dict['drop'])
# for key, value in vars(args).items():
#     print('{}\t{}'.format(key, value))
# print('*' * 50)
# for idx, (key, value) in enumerate(model_dict.items()):
#     for k, v in value.items():
#         print(idx + 1, key, k, v)
# print('*' * 50)
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# orig_train_dataloader = orig_da_train_data_loader(args=args)
# orig_dev_dataloader = orig_da_dev_data_loader(args=args)
#
# drop_train_dataloader = drop_da_train_data_loader(args=args)
# drop_dev_data_loader = drop_da_dev_data_loader(args=args)
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# seed_everything(seed=args.seed)
# if args.orig_model_name is not None:
#     orig_model = load_pretrained_model(args=args, pretrained_model_name=args.orig_model_name)
#     affinity_metrics = affinity_metrics_computation(model=orig_model, dev_data_loader=orig_dev_dataloader,
#                                                     drop_dev_data_loader=drop_dev_data_loader, args=args)
# else:
#     affinity_metrics = 0.0
#
# if args.drop_model_name is not None and args.orig_model_name is not None:
#     drop_model = load_pretrained_model(args=args, pretrained_model_name=args.drop_model_name)
#     orig_model = load_pretrained_model(args=args, pretrained_model_name=args.orig_model_name)
#     diversity_metrics = diversity_metrics_computation(model=orig_model, drop_model=drop_model,
#                                                       train_data_loader=orig_train_dataloader,
#                                                       drop_train_data_loader=drop_train_dataloader,
#                                                       args=args)
# else:
#     diversity_metrics = 0.0
#
# for key, value in vars(args).items():
#     print(key, value)
# print('Affinity metrics: {}'.format(affinity_metrics))
# print('Diversity metrics: {}'.format(diversity_metrics))