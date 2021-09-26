from data_utils.argument_parser import data_aug_default_parser, complete_default_parser
from toyexample_experiments import seed_everything, dev_data_loader
from data_utils.model_utils import model_evaluation
import torch
from data_utils.model_utils import load_pretrained_model
from envs import OUTPUT_FOLDER
from tqdm import tqdm, trange
from os.path import join
import os
from data_utils.da_metrics_utils import MODEL_NAMES, MODEL_500_NAMES, MODEL_1000_NAMES
from data_utils.da_metrics_utils import DROP_MODEL_NAMES
model_dict = {_[0]: _[1] for _ in MODEL_NAMES}
model_500_dict = {_[0]: _[1] for _ in MODEL_500_NAMES}
model_1000_dict = {_[0]: _[1] for _ in MODEL_1000_NAMES}
drop_model_dict = {DROP_MODEL_NAMES[i][0]: DROP_MODEL_NAMES[i][1] for i in range(1, len(DROP_MODEL_NAMES))}

def accuracy_collection(args, trained_model_dict: dict):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # if args.exp_name is None:
    args.exp_name = args.train_file_name + '.models'
    model_name_dict = trained_model_dict[args.exp_name]
    args.exp_name = join(OUTPUT_FOLDER, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    orig_model_name = join(args.exp_name, model_name_dict['orig'])
    beta_drop_model_name = join(args.exp_name, model_name_dict['beta_drop'])
    drop_model_name = join(args.exp_name, model_name_dict['drop'])
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 50)
    for idx, (key, value) in enumerate(trained_model_dict.items()):
        for k, v in value.items():
            print(idx + 1, key, k, v)
    print('*' * 50)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dev_dataloader = dev_data_loader(args=args)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    seed_everything(seed=args.seed)
    orig_model = load_pretrained_model(args=args, pretrained_model_name=orig_model_name)
    orig_model = orig_model.to(args.device)
    orig_acc = model_evaluation(model=orig_model, data_loader=dev_dataloader, args=args)
    drop_model = load_pretrained_model(args=args, pretrained_model_name=drop_model_name)
    drop_model = drop_model.to(args.device)
    drop_acc = model_evaluation(model=drop_model, data_loader=dev_dataloader, args=args)
    beta_drop_model = load_pretrained_model(args=args, pretrained_model_name=beta_drop_model_name)
    beta_drop_model = beta_drop_model.to(args.device)
    beta_acc = model_evaluation(model=beta_drop_model, data_loader=dev_dataloader, args=args)
    return orig_acc, drop_acc, beta_acc
    # return torch.as_tensor(0), torch.as_tensor(0), torch.as_tensor(0)


def dropratio_accuracy_collection(args, drop_ratio, drop_trained_model_dict: dict):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # if args.exp_name is None:
    args.exp_name = DROP_MODEL_NAMES[0] + '.models'
    model_name_dict = drop_trained_model_dict[drop_ratio]
    args.exp_name = join(OUTPUT_FOLDER, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    orig_model_name = join(args.exp_name, model_name_dict['orig'])
    beta_drop_model_name = join(args.exp_name, model_name_dict['beta_drop'])
    drop_model_name = join(args.exp_name, model_name_dict['drop'])
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 50)
    for idx, (key, value) in enumerate(drop_trained_model_dict.items()):
        for k, v in value.items():
            print(idx + 1, key, k, v)
    print('*' * 50)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dev_dataloader = dev_data_loader(args=args)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    seed_everything(seed=args.seed)
    orig_model = load_pretrained_model(args=args, pretrained_model_name=orig_model_name)
    orig_model = orig_model.to(args.device)
    orig_acc = model_evaluation(model=orig_model, data_loader=dev_dataloader, args=args)
    drop_model = load_pretrained_model(args=args, pretrained_model_name=drop_model_name)
    drop_model = drop_model.to(args.device)
    drop_acc = model_evaluation(model=drop_model, data_loader=dev_dataloader, args=args)
    beta_drop_model = load_pretrained_model(args=args, pretrained_model_name=beta_drop_model_name)
    beta_drop_model = beta_drop_model.to(args.device)
    beta_acc = model_evaluation(model=beta_drop_model, data_loader=dev_dataloader, args=args)
    return orig_acc, drop_acc, beta_acc


if __name__ == '__main__':
    # train_file_names = ['train_fastsingle_cat_100_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_200_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_500_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_1000_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_2000_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_5000_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_10000_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_20000_42_300_0.5.pkl.gz']

    # train_file_names = ['train_fastsingle_cat_500_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_1000_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_2000_42_300_0.5.pkl.gz']

    # train_file_names = ['train_fastsingle_cat_5000_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_10000_42_300_0.5.pkl.gz',
    #                     'train_fastsingle_cat_20000_42_300_0.5.pkl.gz']

    train_file_names = ['train_fastsingle_cat_1000_42_300_0.5.pkl.gz',
                        'train_fastsingle_cat_1000_43_300_0.5.pkl.gz',
                        'train_fastsingle_cat_1000_44_300_0.5.pkl.gz',
                        'train_fastsingle_cat_1000_45_300_0.5.pkl.gz',
                        'train_fastsingle_cat_1000_46_300_0.5.pkl.gz']

    # dev_file_names = ['eval_fastsingle_cat_10000_2345_325_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_315_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_310_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_305_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_300_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_295_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_290_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_285_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_280_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_275_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_270_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_265_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_260_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_255_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_250_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_225_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_200_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_175_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_150_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_125_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_100_0.5.pkl.gz',
    #                   'eval_fastsingle_cat_10000_2345_50_0.5.pkl.gz']
    dev_file_names = [
                      'eval_fastsingle_cat_10000_2345_270_0.5.pkl.gz']
    parser = data_aug_default_parser()
    args = parser.parse_args()
    args = complete_default_parser(args=args)

    accuracy_list = []
    for train_file_name in train_file_names:
        args.train_file_name = train_file_name
        accuracy_sub_list = []
        print(train_file_name)
        for eval_file_name in dev_file_names:
            args.eval_file_name = eval_file_name
            orig_acc, drop_acc, beta_acc = accuracy_collection(args=args, trained_model_dict=model_1000_dict)
            res = (eval_file_name, orig_acc.data.item(), drop_acc.data.item(), beta_acc.data.item())
            accuracy_sub_list.append(res)
            print(res)
        accuracy_list.append((train_file_name, accuracy_sub_list))

    # accuracy_list = []
    # drop_ratios = [0.2, 0.3, 0.5, 0.75]
    # for drop_ratio in drop_ratios:
    #     accuracy_sub_list = []
    #     print(drop_ratio)
    #     for eval_file_name in dev_file_names:
    #         args.eval_file_name = eval_file_name
    #         orig_acc, drop_acc, beta_acc = dropratio_accuracy_collection(args=args, drop_ratio=drop_ratio)
    #         res = (eval_file_name, orig_acc.data.item(), drop_acc.data.item(), beta_acc.data.item())
    #         accuracy_sub_list.append(res)
    #         print(res)
    #     accuracy_list.append((drop_ratio, accuracy_sub_list))

    for _ in accuracy_list:
        print(_[0])
        for x in _[1]:
            print(x)
        print('*' * 75)