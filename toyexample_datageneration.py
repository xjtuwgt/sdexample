from data_utils.findcat import FindCatDataset
from envs import HOME_DATA_FOLDER
from os.path import join
import os
import argparse

animal_vocab = ['ant', 'bear', 'cat', 'dog', 'eagle', 'fox', 'goat', 'horse',
                'indri', 'jaguar', 'koala', 'lion', 'moose', 'numbat', 'otter',
                'pig', 'quail', 'rabbit', 'shark', 'tiger', 'uguisu', 'wolf',
                'xerus', 'yak', 'zebra']

test_animals = ['snake', 'robin', 'puma', 'oyster', 'ibis']

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def zero_shot_split():
    train_animal_targets = '_'.join(animal_vocab)
    test_animal_targets = '_'.join(test_animals)
    return train_animal_targets, test_animal_targets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default=join(HOME_DATA_FOLDER, 'toy_data'),
                        help='Directory to save row_data')
    parser.add_argument('--multi_target', type=str, default='single')
    parser.add_argument('--train_data_size', type=int, default=500, help='train data size')
    parser.add_argument('--train_pos_label_ratio', type=float, default=0.5, help='label distribution')
    parser.add_argument('--train_target_tokens', type=str, default='cat')
    parser.add_argument('--zero_shot', type=boolean_string, default='true')
    parser.add_argument('--train_seq_len', type=str, default='300')
    parser.add_argument('--train_target_position', type=str, default=None)
    parser.add_argument('--train_top_position', type=int, default=None)
    parser.add_argument('--train_seed', type=int, default=42, help='random seed for training data generation')

    parser.add_argument('--test_data_size', type=int, default=10000, help='test data size')
    parser.add_argument('--test_target_tokens', type=str, default='cat')
    parser.add_argument('--test_seq_len', type=str, default='300')
    parser.add_argument('--test_pos_label_ratio', type=float, default=0.5, help='test label distribution')
    parser.add_argument("--test_seed", type=int, default=1234, help='random seed for testing data generation')

    parser.add_argument('--eval_data_size', type=int, default=10000, help='eval data size')
    parser.add_argument('--eval_seq_len', type=str, default='300')
    parser.add_argument('--eval_pos_label_ratio', type=float, default=0.5, help='eval label distribution')
    parser.add_argument("--eval_seed", type=int, default=2345, help='random seed for evaluation data generation')

    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    args = parser.parse_args()
    for key, value in vars(args).items():
        print('Parameter: {}\t{}'.format(key, value))
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    os.makedirs(args.data_dir, exist_ok=True)
    train_seq_len = args.train_seq_len
    train_seq_len = tuple([int(x) for x in train_seq_len.split(',')])
    top_position = args.train_top_position
    target_position = args.train_target_position
    if target_position is not None:
        target_position = tuple([int(x) for x in target_position.split(',')])

    if args.zero_shot:
        train_targets, test_targets = zero_shot_split()
        args.train_target_tokens = train_targets
        args.test_target_tokens = test_targets
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_data_set = FindCatDataset(total_examples=args.train_data_size,
                                    target_tokens=args.train_target_tokens,
                                    seqlen=train_seq_len,
                                    prob=args.train_pos_label_ratio,
                                    fixed_positions=target_position,
                                    top_position=top_position,
                                    multi_target=args.multi_target in ['multi'],
                                    seed=args.train_seed)
    train_data_file_name = join(args.data_dir, 'train_' + args.multi_target + '_' + args.train_target_tokens + '_' + str(args.train_data_size)
         + '_' + str(args.train_seed) + '_' + str(args.train_seq_len) + '_' +
         str(args.train_pos_label_ratio))
    if target_position is not None:
        train_data_file_name = train_data_file_name + '_' + args.train_target_position
    if top_position is not None:
        train_data_file_name = train_data_file_name + '_' + str(args.train_top_position)
    train_data_file_name = train_data_file_name + '.pkl.gz'
    train_data_set.save_data_into_file(data_file_name=train_data_file_name)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # test_seq_len = args.test_seq_len
    # test_seq_len = tuple([int(x) for x in test_seq_len.split(',')])
    # test_data_set = FindCatDataset(total_examples=args.test_data_size,
    #                                target_tokens=args.test_target_tokens,
    #                                seqlen=test_seq_len,
    #                                multi_target=args.multi_target in ['multi'],
    #                                seed=args.test_seed)
    # test_data_file_name = join(args.data_dir, 'test_' + args.multi_target + '_' + args.test_target_tokens + '_' + str(args.test_data_size)
    #                             + '_' + str(args.test_seed) + '_' + args.test_seq_len + '_' +
    #                            str(args.test_pos_label_ratio) + '.pkl.gz')
    # test_data_set.save_data_into_file(data_file_name=test_data_file_name)
    # #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # eval_seq_len = args.eval_seq_len
    # eval_seq_len = tuple([int(x) for x in eval_seq_len.split(',')])
    # eval_data_set = FindCatDataset(total_examples=args.eval_data_size,
    #                                target_tokens=args.test_target_tokens,
    #                                seqlen=eval_seq_len,
    #                                multi_target=args.multi_target in ['multi'],
    #                                seed=args.eval_seed)
    # eval_data_file_name = join(args.data_dir, 'eval_' + args.multi_target + '_' + args.test_target_tokens + '_' + str(args.eval_data_size)
    #                             + '_' + str(args.eval_seed) + '_' + args.eval_seq_len + '_' +
    #                            str(args.eval_pos_label_ratio) + '.pkl.gz')
    # eval_data_set.save_data_into_file(data_file_name=eval_data_file_name)