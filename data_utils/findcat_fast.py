from dataclasses import dataclass, field
import numpy as np
import random
from tqdm import tqdm
import torch
from typing import List, Tuple

import gzip
import pickle

from data_utils.dataset import TokenizedDataset, SentenceDropDataset
from data_utils.sentence import Sentence
from data_utils.example import ExampleWithSentences

##random.sample vs random.choice

@dataclass
class FindCatSentence(Sentence):
    pass


@dataclass
class FindCatExample(ExampleWithSentences):
    target_tokens: List[int]
    positions: List[int]
    label: int = 0

def contains_subsequence(target, sequence):
    if len(target) == 0:
        return True
    remaining = sequence
    matched = 0
    for t in target:
        idx = 0
        while idx < len(remaining) and remaining[idx] != t:
            idx += 1
        if idx >= len(remaining):
            return False
        else:
            matched += 1
            if matched == len(target):
                return True
            remaining = remaining[idx + 1:]


RESERVED_TOKENS = 10
VOCAB_SIZE = 26
PAD = 0
CLS = 1
SEP = 2
MASK = 3
VOCAB = tuple(range(RESERVED_TOKENS, RESERVED_TOKENS + VOCAB_SIZE))

def pos_count_over_V_n_generation(target_tokens, vocab, target_tokens_array, exam_seq_len):
    V = len(vocab)
    pos_count_over_V_n = np.zeros((exam_seq_len + 1, len(target_tokens) + 1))
    for n in range(1, exam_seq_len + 1):
        for m in range(1, min(n + 1, max(len(t) for t in target_tokens_array) + 1)):
            if m == n:
                pos_count_over_V_n[n, m] = 1 - V ** (-n)
            elif m == 1:
                pos_count_over_V_n[n, m] = ((V - 1) / V) ** n
            else:
                pos_count_over_V_n[n, m] = (pos_count_over_V_n[n - 1, m - 1] + (V - 1) * pos_count_over_V_n[
                    n - 1, m]) / V
    return pos_count_over_V_n

def neg_example_generation(target_tokens, vocab, exam_seq_len, pos_count_over_V_n):
    retval = []
    matched = 0
    V = len(vocab)
    for i in range(exam_seq_len):
        n = exam_seq_len - i
        m = len(target_tokens) - matched
        if m > n:
            # remaining target is shorter than remaining whole sequence
            retval.append(random.choice(vocab))
        else:
            match_weight = pos_count_over_V_n[n-1, m-1]
            unmatch_weight = pos_count_over_V_n[n-1, m]
            p = np.full(V, unmatch_weight)
            p[vocab.index(target_tokens[matched])] = match_weight
            retval.extend(random.choices(vocab, weights=p))
        if retval[-1] == target_tokens[matched]:
            matched += 1
    assert not contains_subsequence(target_tokens, retval)
    return retval

def pos_count_over_V_n_array_generation(target_tokens_array: list, examp_seq_len, vocab):
    pos_count_over_V_n_array = []
    for target_tokens in target_tokens_array:
        pos_count_over_V_n_array.append(pos_count_over_V_n_generation(target_tokens=target_tokens,
                                                                      target_tokens_array=target_tokens_array,
                                                                      vocab=vocab, exam_seq_len=examp_seq_len))
    return pos_count_over_V_n_array

class FindCatDataset(TokenizedDataset):
    def __init__(self, tokenizer_class="bert-base-uncased",
                 total_examples=1000, seqlen=(300,), vocab=VOCAB,
                 target_tokens='cat', prob=0.5,
                 top_position=None,
                 fixed_positions=None,
                 eval=False, multi_target=True, seed=42, data_file_name=None):
        super().__init__(tokenizer_class=tokenizer_class)
        random.seed(seed)

        self.prob = prob
        self.multi_target = multi_target
        self.seqlen = seqlen[0]
        self.vocab = vocab
        self.target_tokens = [[ord(x) - ord('a') + RESERVED_TOKENS for x in y] for y in target_tokens.split('_')]
        self.fixed_positions = fixed_positions
        self.total_examples = total_examples
        self.top_position = top_position
        # self.data = [self._generate_example() for _ in tqdm(range(self.total_examples))]
        self.pos_count_over_V_n_array = pos_count_over_V_n_array_generation(target_tokens_array=self.target_tokens,
                                                                            examp_seq_len=self.seqlen, vocab=vocab)
        # print('I am here')
        if data_file_name is None:
            self.data = self.data_generation()
        else:
            self.data = self.load_data_from_file(data_file_name=data_file_name)

    def _generate_example(self):
        target = int(random.random() < self.prob)
        target_tokens_idx = random.choice(list(range(len(self.target_tokens))))
        target_tokens = self.target_tokens[target_tokens_idx]
        ##=========
        exam_seq_len = self.seqlen
        pos_count_over_V_n = self.pos_count_over_V_n_array[target_tokens_idx]
        ##=========
        retval = neg_example_generation(target_tokens=target_tokens, exam_seq_len=exam_seq_len,
                                        pos_count_over_V_n=pos_count_over_V_n, vocab=self.vocab)
        positions = []
        if target == 1:
            if self.fixed_positions is not None:
                assert len(self.fixed_positions) == len(target_tokens)
                positions = self.fixed_positions
            else:
                if self.top_position is None:
                    # positions = sorted(random.choices(list(range(exam_seq_len)), k=len(target_tokens)))
                    positions = sorted(random.sample(list(range(exam_seq_len)), k=len(target_tokens)))
                else:
                    top_len = self.top_position if self.top_position < exam_seq_len else exam_seq_len
                    # positions = sorted(random.choices(list(range(top_len)), k=len(target_tokens)))
                    positions = sorted(random.sample(list(range(top_len)), k=len(target_tokens)))

            for p_i, p in enumerate(positions):
                retval[p] = target_tokens[p_i]

        return FindCatExample(
            tokenized_sentences=[FindCatSentence(sentence_idx=s_i, token_ids=[s]) for s_i, s in enumerate(retval)],
            target_tokens=target_tokens, positions=positions, label=target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def data_generation(self):
        data = [self._generate_example() for _ in tqdm(range(self.total_examples))]
        return data

    def save_data_into_file(self, data_file_name):
        with gzip.open(data_file_name, 'wb') as fout:
            pickle.dump(self.data, fout)
        print('save {} records into {}'.format(len(self.data), data_file_name))

    def load_data_from_file(self, data_file_name):
        with gzip.open(data_file_name, 'rb') as fin:
            data = pickle.load(fin)
        print('load {} records from {}'.format(len(data), data_file_name))
        return data


def find_cat_collate_fn(examples):
    ex_lens = [3 + len(ex.target_tokens) + len(ex.tokenized_sentences) for ex in examples]
    max_ex_len = max(ex_lens)

    batched_input = np.full((len(examples), max_ex_len), -1, dtype=np.int64)
    batched_labels = np.zeros((len(examples),), dtype=np.int64)

    for ex_i, ex in enumerate(examples):
        batched_input[ex_i, :ex_lens[ex_i]] = [CLS] + ex.target_tokens + [SEP] + [s.token_ids[0] for s in
                                                                                  ex.tokenized_sentences] + [SEP]
        batched_labels[ex_i] = ex.label

    retval = {
        'input': batched_input,
        'labels': batched_labels
    }
    retval = {k: torch.from_numpy(retval[k]) for k in retval}
    return retval


def find_cat_validation_fn(ex):
    return (ex.label == 0) or contains_subsequence(ex.target_tokens, [s.token_ids[0] for s in ex.tokenized_sentences])

# if __name__ == "__main__":
#
#     # dataset = FindCatDataset(total_examples=10)
#     #
#     # examples = dataset.data
#     #
#     # print(len(examples))
#     # # for x in examples:
#     # #     print(len(x.tokenized_sentences))
#     #
#     cached_examples_file = 'test.pkl.gz'
#     # # with gzip.open(cached_examples_file, 'wb') as fout:
#     # #     pickle.dump(examples, fout)
#     # # print('save {} records into {}'.format(len(examples), cached_examples_file))
#     #
#     # dataset.save_data_into_file(data_file_name=cached_examples_file)
#
#     dataset = FindCatDataset(data_file_name=cached_examples_file)
#
#
#     # true_count = 0
#     # count = 20000
#     # for _ in tqdm(range(count)):
#     #     x = random.choices(dataset.vocab, k=300)
# #         y = contains_subsequence(target=dataset.target_tokens, sequence=x)
# #         if y:
# #             true_count = true_count + 1
# #         # while y:
# #         #     x = random.choices(dataset.vocab, k=300)
# #         #     y = contains_subsequence(target=dataset.target_tokens, sequence=x)
# #         #     true_count = true_count + 1
# #     print(true_count * 1.0/count)
# #
# #     # dataset.data_generation()
# #     # sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=.1,
# #     #                                     example_validate_fn=lambda ex: find_cat_validation_fn(ex, dataset=dataset))
# #     #
# #     # from tqdm import tqdm
# #     # from torch.utils.data import DataLoader
# #     #
# #     # dataloader = DataLoader(sdrop_dataset, batch_size=32, collate_fn=find_cat_collate_fn)
# #     #
# #     # for batch in tqdm(dataloader):
#     #     pass