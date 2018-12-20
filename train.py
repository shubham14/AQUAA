from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pdb, pprint, argparse, json
import warnings
import tensorflow as tf
from os.path import join as pjoin
import numpy as np
from utils.read_data import *
from config import Config as cfg
import time
from model import *
import logging

logging.basicConfig(level=logging.INFO)

# supress warning messages
warnings.filterwarnings('ignore')

np.random.seed(0)

def parse_arg():
    parser = argparse.ArgumentParser(description='train qa model')
    parser.add_argument('--valohai', dest='valohai', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--restore', dest='restore', action='store_true')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=2e-3, help='learning_rate')
    parser.add_argument('--keep_prob', dest='keep_prob', type=float, default=0.9, help='keep prob of dropout')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='batch size to use during training.')
    parser.add_argument('--embed_size', dest='embed_size', type=int, default=100, help='size of pretrained vocab')
    parser.add_argument('--state_size', dest='state_size', type=int, default=64, help='size of each model layer')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='adam')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='outputs', help='directory of outputs')
    parser.add_argument('--reg', dest='reg', type=float, default=0.001, help='rate of regularization')
    args = parser.parse_args()
    return args


def update_config(args, c_time):
    '''update the configuration'''
    if args.valohai:
        print(json.dumps('using valohai mode'))
        cfg.valohai = True
        cfg.output = os.getenv('VH_OUTPUTS_DIR', '/valohai/outputs')
    else:
        cfg.output = args.output_dir
    if args.restore:
        cfg.output = cfg.output + '/output_%s' % c_time

    cfg.start_lr = args.learning_rate
    cfg.keep_prob = args.keep_prob
    cfg.batch_size = args.batch_size
    cfg.embed_size = args.embed_size
    cfg.lstm_num_hidden = args.state_size
    cfg.opt = args.optimizer
    cfg.reg = args.reg
    cfg.train_dir = pjoin(cfg.output, cfg.train_dir)
    cfg.log_dir = pjoin(cfg.output, cfg.log_dir)
    cfg.cache_dir = cfg.output + cfg.cache_dir
    cfg.summary_dir = cfg.output + cfg.summary_dir


def initialize_model(session, model, train_dir):
    '''
    Initialiize model parameters 
    '''
    ckpt = tf.train.get_checkpoint_state(train_dir)
    print(train_dir, ckpt)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model



def initialize_vocab(vocab_path):
    '''
    Create a vocab dictionary for mapping words to embeddings
    '''    
    # vocab_path = 'amazon/vocab.dat'
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, "rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip(b'\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)



def main(_):
    c_time = time.strftime('%Y%m%d_%H%M', time.localtime())
    args = parse_arg()
    update_config(args, c_time)
    
    # logging the configurations for display
    logging.info(cfg)
    if args.test:
        pdb.set_trace()

    data_dir = cfg.DATA_DIR
    set_names = cfg.set_names
    suffixes = cfg.suffixes
    dataset = mask_dataset(data_dir, set_names, suffixes)
    answers = read_answers(data_dir)
    raw_answers = read_raw_answers(data_dir)

    vocab_path = pjoin(data_dir, cfg.vocab_file)
    vocab, rev_vocab = initialize_vocab(vocab_path)

    # gpu setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    encoder = Encoder(size=2 * cfg.lstm_num_hidden)
    decoder = Decoder(output_size=2 * cfg.lstm_num_hidden)
    qa = QASystem(encoder, decoder, cfg.embed_dir)

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        initialize_model(sess, qa, cfg.train_dir)
        if args.test:
            qa.train(cfg.start_lr, sess, dataset, answers, cfg.train_dir,
                     raw_answers=raw_answers,
                     debug_num=100,
                     rev_vocab=rev_vocab)
        else:
            qa.train(cfg.start_lr, sess, dataset, answers, cfg.train_dir,
                     raw_answers=raw_answers,
                     rev_vocab=rev_vocab)
        qa.evaluate_answer(sess, dataset, raw_answers, rev_vocab,
                           log=True,
                           training=True,
                           sample=4000)


if __name__ == "__main__":
    tf.app.run()
