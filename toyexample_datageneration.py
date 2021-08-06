from data_utils.findcat import FindCatDataset
from envs import HOME_DATA_FOLDER
from os.path import join
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default=join(HOME_DATA_FOLDER, 'toy_data'),
                        help='Directory to save row_data')
    parser.add_argument('--multi_target', type=str, default='single')
    parser.add_argument('--train_data_size', type=int, default=500, help='train data size')
    parser.add_argument('--train_pos_label_ratio', type=float, default=0.5, help='label distribution')
    parser.add_argument('--target_tokens', type=str, default='cat')
    parser.add_argument('--train_seq_len', type=str, default='300')
    parser.add_argument('--train_seed', type=int, default=42, help='random seed for training data generation')

    parser.add_argument('--test_data_size', type=int, default=10000, help='test data size')
    parser.add_argument('--test_seq_len', type=str, default='300')
    parser.add_argument('--test_pos_label_ratio', type=float, default=0.5, help='test label distribution')
    parser.add_argument("--test_seed", type=int, default=1234, help='random seed for testing data generation')


    args = parser.parse_args()
    for key, value in vars(args).items():
        print('Parameter: {}\t{}'.format(key, value))

    os.makedirs(args.data_dir, exist_ok=True)
    train_seq_len = args.train_seq_len
    train_seq_len = tuple([int(x) for x in train_seq_len.split(',')])
    train_data_set = FindCatDataset(total_examples=args.train_data_size,
                                    target_tokens=args.target_tokens,
                                    seqlen=train_seq_len,
                                    multi_target=args.multi_target in ['multi'],
                                    seed=args.train_seed)
    train_data_file_name = join(args.data_dir, 'train_' + args.multi_target + '_' + args.target_tokens + '_' + str(args.train_data_size)
                                + '_' + str(args.train_seed) + '_' + str(args.train_seq_len) + '_' +
                                str(args.train_pos_label_ratio) + '.pkl.gz')
    train_data_set.save_data_into_file(data_file_name=train_data_file_name)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # test_seq_len = args.test_seq_len
    # test_seq_len = tuple([int(x) for x in test_seq_len.split(',')])
    # test_data_set = FindCatDataset(total_examples=args.test_data_size,
    #                                target_tokens=args.target_tokens,
    #                                seqlen=test_seq_len,
    #                                multi_target=args.multi_target in ['multi'],
    #                                seed=args.test_seed)
    # test_data_file_name = join(args.data_dir, 'test_' + args.multi_target + '_' + args.target_tokens + '_' + str(args.test_data_size)
    #                             + '_' + str(args.test_seed) + '_' + args.test_seq_len + '_' +
    #                            str(args.test_pos_label_ratio) + '.pkl.gz')
    # test_data_set.save_data_into_file(data_file_name=test_data_file_name)