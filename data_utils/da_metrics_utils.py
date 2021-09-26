###Tradeoffs in Data Augmentation: An Empirical Study, https://openreview.net/pdf?id=ZcKPWuhG6wy
from data_utils.model_utils import model_loss_computation, model_evaluation
from torch.utils.data import DataLoader
from os.path import join
from envs import HOME_DATA_FOLDER
from utils.env_utils import seed_everything

from data_utils.findcat import FindCatDataset, find_cat_validation_fn, find_cat_collate_fn
from data_utils.dataset import SentenceDropDataset

MODEL_NAMES = [('train_fastsingle_cat_100_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_50_2_dev_0.5585.pkl',
                                                                      'drop': 'model_False_0.1_800_2_dev_0.5948.pkl',
                                                                      'beta_drop': 'model_True_0.1_800_2_dev_0.6133.pkl'}),
               ('train_fastsingle_cat_200_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_100_4_dev_0.5677.pkl',
                                                                      'drop': 'model_False_0.1_325_4_dev_0.6832.pkl',
                                                                      'beta_drop': 'model_True_0.1_450_4_dev_0.7321.pkl'}),
               ('train_fastsingle_cat_500_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_350_8_dev_0.6249.pkl',
                                                                      'drop': 'model_False_0.1_500_8_dev_0.7750.pkl',
                                                                      'beta_drop': 'model_True_0.1_500_8_dev_0.8518.pkl'}),
               ('train_fastsingle_cat_1000_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_319_12_dev_0.6888.pkl',
                                                                       'drop': 'model_False_0.1_444_12_dev_0.7993.pkl',
                                                                       'beta_drop': 'model_True_0.1_438_8_dev_0.9069.pkl'}),
               ('train_fastsingle_cat_2000_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_385_12_dev_0.7914.pkl',
                                                                       'drop': 'model_False_0.1_444_24_dev_0.8365.pkl',
                                                                       'beta_drop': 'model_True_0.1_444_24_dev_0.9685.pkl'}),
               ('train_fastsingle_cat_5000_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_289_48_dev_0.8920.pkl',
                                                                       'drop': 'model_False_0.1_150_29_dev_0.8287.pkl',
                                                                       'beta_drop': 'model_True_0.1_288_27_dev_0.9849.pkl'}),
               ('train_fastsingle_cat_10000_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_379_154_dev_0.9441.pkl',
                                                                        'drop': 'model_False_0.1_268_81_dev_0.8390.pkl',
                                                                        'beta_drop': 'model_True_0.1_276_25_dev_0.9953.pkl'}),
               ('train_fastsingle_cat_20000_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_365_168_dev_0.9797.pkl',
                                                                        'drop': 'model_False_0.1_454_11_dev_0.8711.pkl',
                                                                        'beta_drop': 'model_True_0.1_488_169_dev_0.9992.pkl'})]

MODEL_500_NAMES = [
               ('train_fastsingle_cat_500_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_350_8_dev_0.6249.pkl',
                                                                      'drop': 'model_False_0.1_500_8_dev_0.7750.pkl',
                                                                      'beta_drop': 'model_True_0.1_500_8_dev_0.8518.pkl'}),
               ('train_fastsingle_cat_500_43_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_375_8_dev_0.5770.pkl',
                                                           'drop': 'model_False_0.1_375_8_dev_0.7201.pkl',
                                                           'beta_drop': 'model_True_0.1_375_8_dev_0.8289.pkl'}),
               ('train_fastsingle_cat_500_44_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_213_4_dev_0.6202.pkl',
                                                           'drop': 'model_False_0.1_388_4_dev_0.7675.pkl',
                                                           'beta_drop': 'model_True_0.1_300_8_dev_0.8502.pkl'}),
               ('train_fastsingle_cat_500_45_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_213_4_dev_0.5795.pkl',
                                                           'drop': 'model_False_0.1_363_4_dev_0.7111.pkl',
                                                           'beta_drop': 'model_True_0.1_363_4_dev_0.8194.pkl'}),
                ('train_fastsingle_cat_500_46_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_338_4_dev_0.6250.pkl',
                                                           'drop': 'model_False_0.1_400_8_dev_0.7483.pkl',
                                                           'beta_drop': 'model_True_0.1_400_8_dev_0.8384.pkl'}),
                ]

MODEL_1000_NAMES = [
               ('train_fastsingle_cat_1000_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_319_12_dev_0.6888.pkl',
                                                                      'drop': 'model_False_0.1_444_12_dev_0.7993.pkl',
                                                                      'beta_drop': 'model_True_0.1_438_8_dev_0.9069.pkl'}),
               ('train_fastsingle_cat_1000_43_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_263_8_dev_0.6667.pkl',
                                                           'drop': 'model_False_0.1_225_16_dev_0.8019.pkl',
                                                           'beta_drop': 'model_True_0.1_225_16_dev_0.8784.pkl'}),
               ('train_fastsingle_cat_1000_44_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_307_4_dev_0.6953.pkl',
                                                           'drop': 'model_False_0.1_382_4_dev_0.7885.pkl',
                                                           'beta_drop': 'model_True_0.1_400_16_dev_0.9027.pkl'}),
               ('train_fastsingle_cat_1000_45_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_388_8_dev_0.6729.pkl',
                                                           'drop': 'model_False_0.1_88_8_dev_0.7415.pkl',
                                                           'beta_drop': 'model_True_0.1_394_12_dev_0.8629.pkl'}),
                ('train_fastsingle_cat_1000_46_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_375_16_dev_0.6879.pkl',
                                                           'drop': 'model_False_0.1_382_4_dev_0.7920.pkl',
                                                           'beta_drop': 'model_True_0.1_332_4_dev_0.8944.pkl'}),
                ]

MODEL_1000_EVAL_NAMES = [
               ('train_fastsingle_cat_1000_42_300_0.5.pkl.gz.models', {'drop': 'model_True_0.1_119_12_mask_False_val_true_42_dev_0.8236.pkl',
                                                                      'beta_drop': 'model_True_0.1_450_16_mask_False_val_true_42_dev_0.9206.pkl'}),
               ('train_fastsingle_cat_1000_43_300_0.5.pkl.gz.models', {
                                                           'drop': 'model_False_0.1_388_8_mask_False_val_true_42_dev_0.7789.pkl',
                                                           'beta_drop': 'model_True_0.1_488_8_mask_False_val_true_42_dev_0.8955.pkl'}),
               ('train_fastsingle_cat_1000_44_300_0.5.pkl.gz.models', {
                                                           'drop': 'model_False_0.1_388_8_mask_False_val_true_42_dev_0.8063.pkl',
                                                           'beta_drop': 'model_True_0.1_350_16_mask_False_val_true_42_dev_0.9156.pkl'}),
               ('train_fastsingle_cat_1000_45_300_0.5.pkl.gz.models', {
                                                           'drop': 'model_False_0.1_475_16_mask_False_val_true_42_dev_0.7914.pkl',
                                                           'beta_drop': 'model_True_0.1_463_8_mask_False_val_true_42_dev_0.8961.pkl'})]

DROP_MODEL_NAMES = ['train_fastsingle_cat_500_42_300_0.5.pkl.gz', (0.2, {'orig': 'model_False_0.0_350_8_dev_0.6249.pkl',
                           'drop': 'model_False_0.2_263_4_dev_0.6755.pkl',
                           'beta_drop': 'model_True_0.2_338_4_dev_0.8267.pkl'}),
                    (0.3, {'orig': 'model_False_0.0_350_8_dev_0.6249.pkl',
                           'drop': 'model_False_0.3_350_8_dev_0.6366.pkl',
                           'beta_drop': 'model_True_0.3_338_4_dev_0.7938.pkl'}),
                    (0.5, {'orig': 'model_False_0.0_350_8_dev_0.6249.pkl',
                           'drop': 'model_False_0.5_88_4_dev_0.5917.pkl',
                           'beta_drop': 'model_True_0.5_363_4_dev_0.7812.pkl'}),
                    (0.75, {'orig': 'model_False_0.0_350_8_dev_0.6249.pkl',
                           'drop': 'model_False_0.75_113_4_dev_0.5827.pkl',
                           'beta_drop': 'model_True_0.75_488_4_dev_0.6345.pkl'})]

def affinity_metrics_computation(model, dev_data_loader, drop_dev_data_loader, args):
    """
    :param model: train over clean data
    :param dev_data_loader:
    :param drop_dev_data_loader:
    :return:
    Affinity: Acc(model, drop_dev) / Acc(model, dev)
    """
    model = model.to(args.device)
    drop_acc_list = []
    for i in range(10):
        seed_everything(seed=i)
        drop_acc = model_evaluation(model=model, data_loader=drop_dev_data_loader, args=args)
        drop_acc_list.append(drop_acc.data.item())
    acc = model_evaluation(model=model, data_loader=dev_data_loader, args=args)
    acc = acc.data.item()
    print('Drop accuracy: {}, orig accuracy = {}'.format(drop_acc_list, acc))
    drop_acc = sum(drop_acc_list)/len(drop_acc_list)
    # print(drop_acc_list)
    # affinity = sum([drop_acc/acc for drop_acc in drop_acc_list])/len(drop_acc_list)
    affinity = drop_acc/acc
    return affinity

def diversity_metrics_computation(model, train_data_loader, drop_model, drop_train_data_loader, args):
    """
    :param model: model trained over clean data
    :param train_data_loader:
    :param drop_model: model trained over augmentation data
    :param drop_train_data_loader:
    :return:
    Diversity: Loss(drop_model, drop_train) / Loss(model, train)
    """
    drop_model = drop_model.to(args.device)
    drop_loss_list = []
    for i in range(10):
        drop_loss = model_loss_computation(model=drop_model, data_loader=drop_train_data_loader, args=args)
        drop_loss_list.append(drop_loss)
    model = model.to(args.device)
    loss = model_loss_computation(model=model, data_loader=train_data_loader, args=args)
    drop_loss = sum(drop_loss_list)/len(drop_loss_list)
    print(drop_loss_list)
    print(loss)
    diversity  = drop_loss / loss
    # diversity = sum([drop_loss / loss for drop_loss in drop_loss_list])/len(drop_loss_list)
    return diversity


def orig_da_train_data_loader(args):
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
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=args.batch_size,
                            collate_fn=find_cat_collate_fn)
    print('Original training data loader')
    return dataloader

def drop_da_train_data_loader(args):
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
                            shuffle=False,
                            batch_size=args.batch_size,
                            collate_fn=find_cat_collate_fn)
    print('Training data loader with sentdrop')
    return dataloader

def orig_da_dev_data_loader(args):
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
    dev_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=find_cat_collate_fn)
    print('Original dev data loader')
    return dev_dataloader

def drop_da_dev_data_loader(args):
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
    validation_fn = find_cat_validation_fn if args.validate_examples else lambda ex: True
    sdrop_dataset = SentenceDropDataset(dataset=dataset,
                                        sent_drop_prob=args.sent_dropout,
                                        beta_drop=args.beta_drop,
                                        mask=args.mask,
                                        mask_id=args.mask_id,
                                        example_validate_fn=validation_fn)
    dataloader = DataLoader(sdrop_dataset,
                            shuffle=False,
                            batch_size=args.batch_size,
                            collate_fn=find_cat_collate_fn)
    print('Dev data loader with sentdrop')
    return dataloader

