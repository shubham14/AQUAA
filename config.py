from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from os.path import join as pjoin 

# config file storing the parameters for running the data
class Config:

    context_max_len = 400
    question_max_len = 30
    ROOT_DIR = os.path.dirname(__file__)
    output = 'outputs'
    DATA_DIR = 'data/amazon'
    train_dir = 'ckpt'
    log_dir = 'log'
    fig_dir = '/fig'
    cache_dir = '/cache'
    vocab_file = 'vocab.dat'
    set_names = ['train', 'val']
    suffixes = ['context', 'question']
    lstm_num_hidden = 64
    embed_size = 100
    batch_size = 8
    epochs = 1
    embed_dir = 'data/amazon/glove.trimmed.100.npz'
    max_grad_norm = 10.0
    start_lr = 2e-3
    clip_by_val = 10.
    keep_prob = 0.9
    dtype = tf.float32
    opt = 'adam'
    reg = 0.001
    print_every = 20
    summary_dir = '/tensorboard'
    sample = 100
    save_every = 2000
    save_every_epoch = True
    valohai = False
    model_paths = ['train/ensemble/' + i for i in ['m1', 'm2', 'm3', 'm4']]
    num_eval = 4000
