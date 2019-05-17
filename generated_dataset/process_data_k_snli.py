# Copyright (c) Microsoft. All rights reserved.
import os
import json
import tqdm
import pickle
import re
import collections
import argparse
import csv
from sys import path
from data_utils.vocab import Vocabulary
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_utils.log_wrapper import create_logger
from data_utils.label_map import GLOBAL_MAP
from data_utils.glue_utils import *
DEBUG_MODE=False
MAX_SEQ_LEN = 512

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_snli(file, label_dict, header=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 10
            if blocks[-1] == '-': continue
            lab = label_dict[blocks[-1]]
            if lab is None:
                import pdb; pdb.set_trace()
            lab = 0 if lab is None else lab
            sample = {'uid': blocks[0], 'premise': blocks[7], 'hypothesis': blocks[8], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_file(file, label_dict, header=True):
    rows = []
    seen = set()
    cnt = 0
    with open(file, newline='') as f:
        reader =  csv.reader(f)
        for line in reader:
            if header:
                header = False
                continue
            lab = label_dict["entailment"]
            if lab is None:
                import pdb; pdb.set_trace()
            premise = line[1]
            if premise not in seen:
                sample = {'uid': cnt, 'premise': line[2], 'hypothesis': line[3], 'label': lab}
                rows.append(sample)
                seen.add(premise)
                cnt += 1
    return rows

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def build_data(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tolower=True):
    """Build data of sentence pair tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            hypothesis = bert_tokenizer.tokenize(sample['hypothesis'])
            label = sample['label']
            _truncate_seq_pair(premise, hypothesis, max_seq_len - 3)
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis + ['[SEP]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(hypothesis) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))

if __name__ == "__main__":
    # snli_dev_path = "./dev.tsv"
    # snli_dev_data = load_snli(snli_dev_path, GLOBAL_MAP['snli'])
    test_fout = "./snli_test.json"
    test_data = load_file("./contrapositives_test.csv", GLOBAL_MAP['snli'])
    build_data(test_data, test_fout)
    print(test_data[0])

