import os
import sys

sys.path.append('..')
from os.path import join as pjoin
import numpy as np
from config import Config as cfg
import logging

np.random.seed(0)

def read_answers(data_dir, set_names=['train', 'val'], suffix='.span'):
    assert isinstance(set_names, list), 'the type of set_names should be list.'
    assert isinstance(suffix, str), 'the type of set_names should be string.'

    dict = {}
    for sn in set_names:
        data_path = pjoin(data_dir, sn + suffix)
        assert os.path.exists(data_path),\
            'the path {} does not exist, please check again.'.format(data_path)
        with open(data_path, 'r') as fdata:
            answer = [preprocess_answer(line) for line in fdata.readlines()]
        name = sn + '_answer'
        dict[name] = answer

    return dict


def preprocess_answer(string):
    num = list(map(int, string.strip().split(' ')))
    if min(num) >= cfg.context_max_len:
        num[0] = np.random.randint(10, cfg.context_max_len - 50)
        num[1] = num[0] + np.random.randint(0, 20)
    elif max(num) >= cfg.context_max_len:
        num[1] = cfg.context_max_len - 1
        if num[1] < num[0]:
            num[0] = num[1] - 1
    return num


def read_raw_answers(data_dir, set_names=['train', 'val'], suffixes='.answer'):
    dict = {}
    for sn in set_names:
        data_path = pjoin(data_dir, sn + suffixes)
        assert os.path.exists(data_path), \
            'the path {} does not exist, please check again.'.format(data_path)
        with open(data_path, 'r') as fdata:
            raw_answer = [line.strip() for line in fdata.readlines()]
        name = 'raw_' + sn + '_answer'
        dict[name] = raw_answer

    return dict


def mask_dataset(data_dir, set_names=['train', 'val'], suffixes=['context', 'question']):
    dict = {}
    for sn in set_names:
        for suf in suffixes:
            max_len = cfg.context_max_len if suf == 'context' else cfg.question_max_len
            data_path = pjoin(data_dir, sn + '.ids.' + suf)
            with open(data_path, 'r') as fdata:
                raw_data = [list(map(int, line.strip().split(' '))) for line in fdata.readlines()]
            name = sn + '_' + suf
            masked_data = [mask_input(rd, max_len) for rd in raw_data]
            dict[name] = masked_data

    return dict

def proc_max_prob(y1, y2):
    a_s = np.argmax(y1, axis=1) 
    a_e = np.argmax(y2, axis=1)
    return a_s - np.random.randint(40), a_e + np.random.randint(2, 10)

def mask_input(data_list, max_len):
    l = len(data_list)
    mask = [True] * l
    if l > max_len:
        return data_list[:max_len], mask[:max_len]
    else:
        return data_list + [0] * (max_len - l), mask + [False] * (max_len - l)
