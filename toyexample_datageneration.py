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
    parser.add_argument('--train_data_size', type=int, default=1000, help='train data size')
    parser.add_argument('--target_tokens', type=str, default='cat')
    parser.add_argument('--seq_len', type=int, default=300)
    parser.add_argument('--test_data_size', type=int, default=10000, help='test data size')
    parser.add_argument('--train_seed', type=int, default=42, help='random seed for training data generation')
    parser.add_argument("--test_seed", type=int, default=1234, help='random seed for testing data generation')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    train_data_set = FindCatDataset(total_examples=args.train_data_size,
                                    target_tokens=args.target_tokens,
                                    seqlen=args.seq_len,
                                    seed=args.train_seed)
    train_data_file_name = join(args.data_dir, 'train_' + args.target_tokens + '_' + str(args.train_data_size)
                                + '_' + str(args.train_seed) + '_' + args.seq_len + '.pkl.gz')

    train_data_set.save_data_into_file(data_file_name=train_data_file_name)

    test_data_set = FindCatDataset(total_examples=args.test_data_size,
                                   target_tokens=args.target_tokens,
                                   seqlen=args.seq_len,
                                   seed=args.test_seed)
    test_data_file_name = join(args.data_dir, 'test_' + args.target_tokens + '_' + str(args.test_data_size)
                                + '_' + str(args.test_seed) + '_' + args.seq_len + '.pkl.gz')
    test_data_set.save_data_into_file(data_file_name=test_data_file_name)