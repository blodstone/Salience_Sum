#  Copyright (c) Hardy (hardy.oei@gmail.com) 2020.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Checking the output of preprocess_salience_from_pkupu.py against the raw version.
"""
import json
import pickle

import pytest
from pathlib import Path

from typing import Tuple, Dict


@pytest.fixture
def file_path() -> Tuple[Dict[str, Path], Dict[str, Path]]:
    files = {
        'train': Path('../data/bbc_allen/bbc_tagged_summ/train.tsv.tagged'),
        'test': Path('../data/bbc_allen/bbc_tagged_summ/test.tsv.tagged'),
        'val': Path('../data/bbc_allen/bbc_tagged_summ/val.tsv.tagged'),
    }
    raw = {
        'document': Path('../data/bbc-tokenized-segmented-final/restbody'),
        'summary': Path('../data/bbc-tokenized-segmented-final/firstsentence')
    }
    return files, raw

@pytest.fixture
def index():
    json_index = json.load(open(
        '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc_raw/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json'))
    indexes = {
        'test': pickle.load(open('/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc_raw/index/test_final_idx', 'rb')),
        'train': pickle.load(open('/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc_raw/index/train_final_idx', 'rb')),
        'validation': pickle.load(open('/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc_raw/index/val_final_idx', 'rb')),
    }
    return json_index, indexes

def count_word_tsv(line):
    lines = line.split('\n')
    total = 0
    for line in lines:
        if line.strip() == '':
            continue
        src_tagged_seq, tgt_seq = line.split('\t')
        src_seq, salience_seq = zip(*[group.split(u"￨") for group in src_tagged_seq.split()[:400]])
        total += len(tgt_seq)
    return total


def count_word_simple(line):
    return sum([len(l.split()) for l in line.strip().split('\n')])


def test_word_count(file_path: Tuple[Dict[str, Path], Dict[str, Path]]):
    files, raw = file_path
    assert files['train'].is_file()
    assert files['test'].is_file()
    assert files['val'].is_file()
    assert raw['document'].is_dir()
    assert raw['summary'].is_dir()
    train = files['train'].read_text()
    test = files['test'].read_text()
    val = files['val'].read_text()
    total_output_len = count_word_tsv(train) + count_word_tsv(test) + count_word_tsv(val)
    # total_raw_len = sum([count_word_simple(doc.read_text()) for doc in raw['document'].iterdir()]) + sum([count_word_simple(summ.read_text()) for summ in raw['summary'].iterdir()])
    total_raw_len = sum([count_word_simple(summ.read_text()) for summ in raw['summary'].iterdir()])
    assert total_output_len == total_raw_len, f'{total_output_len} != {total_raw_len}'

# def test_file_order(file_path: Tuple[Dict[str, Path], Dict[str, Path]]):
#     files, raw, index = file_path
#     train = files['train'].read_text()
#     test = files['test'].read_text()
#     val = files['val'].read_text()
#     x = sorted([line.split('\t')[1] for line in test.split('\n') + train.split('\n') + val.split('\n') if line.strip() != ''])
#     y = sorted([summ.read_text().strip().lower() for summ in raw['summary'].iterdir()])
#     assert x[0] == y[0]
#     assert len(x) == len(index['train']) + len(index['validation']) + len(index['test'])

def test_index(index):
    json_index, indexes = index
    print()