# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from model import *
from os.path import join as pjoin
from new_data_loader import *
from config import Config as cfg
from train import initialize_vocab, initialize_model
from utils.read_data import mask_input
import nltk
import pickle


def main():
    dataload = DataLoader()
    vocab, rev_vocab = initialize_vocab(pjoin(cfg.DATA_DIR,cfg.vocab_file))
    config = tf.ConfigProto(device_count= {'GPU': 0})
    #config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    encoder = Encoder(size=2 * cfg.lstm_num_hidden)
    decoder = Decoder(output_size=2 * cfg.lstm_num_hidden)
    qa = QASystem(encoder, decoder, cfg.embed_dir)

    c1 = open(pjoin(cfg.DATA_DIR, 'test.context'), 'r').read().split('\n')
    q1 = open(pjoin(cfg.DATA_DIR, 'test.question'), 'r').read().split('\n')
    a1 = open(pjoin(cfg.DATA_DIR, 'test.answer'), 'r').read().split('\n')
    
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        load_train_dir = pjoin(cfg.output, cfg.train_dir)
        initialize_model(sess, qa, load_train_dir)
        ans = []
        f1 = []
        for i, data in enumerate(c1):
            print (i)
            sentence = c1[i]
            query = q1[i]
            raw_context = nltk.word_tokenize(sentence)
            len(raw_context)
            context = dataload.sentence_to_token_ids(sentence, vocab, tokenizer=nltk.word_tokenize)
            question = dataload.sentence_to_token_ids(query, vocab, tokenizer=nltk.word_tokenize)
            context_in = mask_input(context, cfg.context_max_len)
            question_in = mask_input(question, cfg.question_max_len)
            start, end = qa.answer(sess, [context_in], [question_in], train=False)
            answer = ' '.join(raw_context[start[0]: (end[0] + 1)])
            f1.append(qa.f1_score(answer, a1[i]))
            print("QUESTION: " + query)
            print("ANSWER: "+ answer)
            if i == 100:
                break
            ans.append(answer)   
    return ans, f1

if __name__ == "__main__":
    ans, f1 = main()
    mean_f1 = float(sum(f1))/len(f1)
    print (mean_f1)
