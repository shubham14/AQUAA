from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Counter
import time
import logging
import re
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import string
from utils.matchLSTM_cell import matchLSTMcell
import tensorflow.contrib.rnn as rnn
from config import Config as cfg
import sys
from os.path import join as pjoin
from tqdm import tqdm
import warnings
from utils.read_data import *

# surpress warnings
warnings.filterwarnings("ignore")
regularizer = tf.contrib.layers.l2_regularizer(cfg.reg)
dtype = cfg.dtype


logging.basicConfig(level=logging.INFO)

def sequence_length(sequence_mask):
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)

class Encoder(object):
    '''
    Encoder class which contains the matchLSTM for encoding the input sequence 
    of context and questions
    '''
    def __init__(self, vocab_dim=cfg.embed_size, size=2 * cfg.lstm_num_hidden):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, context, context_m, question, question_m, embedding, keep_prob):
        context_embed = tf.nn.embedding_lookup(embedding, context)
        context_embed = tf.nn.dropout(context_embed, keep_prob=keep_prob)
        question_embed = tf.nn.embedding_lookup(embedding, question)
        question_embed = tf.nn.dropout(question_embed, keep_prob=keep_prob)
        with tf.variable_scope('context'):
            con_lstm_fw_cell = rnn.BasicLSTMCell(cfg.lstm_num_hidden, forget_bias=1.0)
            con_lstm_bw_cell = rnn.BasicLSTMCell(cfg.lstm_num_hidden, forget_bias=1.0)
            con_outputs, con_outputs_states = tf.nn.bidirectional_dynamic_rnn(
                con_lstm_fw_cell,
                con_lstm_bw_cell,
                context_embed,
                sequence_length=sequence_length(context_m),
                dtype=dtype, scope='lstm')

        # encoding context
        with tf.name_scope('H_context'):
            H_context = tf.concat(con_outputs, axis=2)
            # TODO: add drop out
            H_context = tf.nn.dropout(H_context, keep_prob=keep_prob)

        with tf.variable_scope('question'):
            ques_lstm_fw_cell = rnn.BasicLSTMCell(cfg.lstm_num_hidden, forget_bias=1.0)
            ques_lstm_bw_cell = rnn.BasicLSTMCell(cfg.lstm_num_hidden, forget_bias=1.0)
            ques_outputs, ques_outputs_states = tf.nn.bidirectional_dynamic_rnn(ques_lstm_fw_cell,
                                                                                ques_lstm_bw_cell,
                                                                                question_embed,
                                                                                sequence_length=sequence_length(
                                                                                    question_m),
                                                                                dtype=dtype, scope='lstm')
        # encoding questions
        with tf.name_scope('H_question'):
            H_question = tf.concat(ques_outputs, 2)
            # TODO: add drop out
            H_question = tf.nn.dropout(H_question, keep_prob=keep_prob)

        # MatchLSTM after concatenation of bi-directional LSTM
        with tf.variable_scope('Hr'):
            matchlstm_fw_cell = matchLSTMcell(2 * cfg.lstm_num_hidden, self.size, H_question,
                                              question_m)
            matchlstm_bw_cell = matchLSTMcell(2 * cfg.lstm_num_hidden, self.size, H_question,
                                              question_m)
            H_r, _ = tf.nn.bidirectional_dynamic_rnn(matchlstm_fw_cell,
                                                     matchlstm_bw_cell,
                                                     H_context,
                                                     sequence_length=sequence_length(context_m),
                                                     dtype=dtype)

        with tf.name_scope('H_r'):
            H_r = tf.concat(H_r, axis=2)
            H_r = tf.nn.dropout(H_r, keep_prob=keep_prob)

        return H_r


class Decoder(object):
    '''
    Decoder class which contains PointerNet Implementation
    assigns probabilities to start and end index of an answer
    given a context and question
    '''
    def __init__(self, output_size=2 * cfg.lstm_num_hidden):
        self.output_size = output_size

    def decode(self, H_r, context_m, keep_prob):
        context_m = tf.cast(context_m, tf.float32)
        initializer = tf.contrib.layers.xavier_initializer()

        shape_Hr = tf.shape(H_r)
        Wr = tf.get_variable('Wr', [4 * cfg.lstm_num_hidden, 2 * cfg.lstm_num_hidden], dtype,
                             initializer, regularizer=regularizer)
        Wh = tf.get_variable('Wh', [4 * cfg.lstm_num_hidden, 2 * cfg.lstm_num_hidden], dtype,
                             initializer, regularizer=regularizer)
        Wf = tf.get_variable('Wf', [2 * cfg.lstm_num_hidden, 1], dtype,
                             initializer, regularizer=regularizer)
        br = tf.get_variable('br', [2 * cfg.lstm_num_hidden], dtype,
                             tf.zeros_initializer())
        bf = tf.get_variable('bf', [1, ], dtype,
                             tf.zeros_initializer())

        wr_e = tf.tile(tf.expand_dims(Wr, axis=[0]), [shape_Hr[0], 1, 1])
        f = tf.tanh(tf.matmul(H_r, wr_e) + br)
        f = tf.nn.dropout(f, keep_prob=keep_prob)

        wf_e = tf.tile(tf.expand_dims(Wf, axis=[0]), [shape_Hr[0], 1, 1])
        
        # scores of start token.
        with tf.name_scope('starter_score'):
            s_score = tf.squeeze(tf.matmul(f, wf_e) + bf, axis=[2])
        
        # for checking out the probabilities of starter index
        with tf.name_scope('starter_prob'):
            s_prob = tf.nn.softmax(s_score)
            s_prob = tf.multiply(s_prob, context_m)

        Hr_attend = tf.reduce_sum(tf.multiply(H_r, tf.expand_dims(s_prob, axis=[2])), axis=1)
        e_f = tf.tanh(tf.matmul(H_r, wr_e) +
                      tf.expand_dims(tf.matmul(Hr_attend, Wh), axis=[1])
                      + br)

        with tf.name_scope('end_score'):
            e_score = tf.squeeze(tf.matmul(e_f, wf_e) + bf, axis=[2])
        # for checking out the probabilities of end index
        with tf.name_scope('end_prob'):
            e_prob = tf.nn.softmax(e_score)
            e_prob = tf.multiply(e_prob, context_m)
        return s_score, e_score


# combines the Encoder-Decoder architecture into a single
# Question answering system
class QASystem(object):
    def __init__(self, encoder, decoder, embed_path):
        self.embed_path = embed_path
        self.max_grad_norm = cfg.max_grad_norm
        self.encoder = encoder
        self.decoder = decoder
        self.context = tf.placeholder(tf.int32, (None, cfg.context_max_len))
        self.context_m = tf.placeholder(tf.bool, (None, cfg.context_max_len))
        self.question = tf.placeholder(tf.int32, (None, cfg.question_max_len))
        self.question_m = tf.placeholder(tf.bool, (None, cfg.question_max_len))
        self.answer_s = tf.placeholder(tf.int32, (None,))
        self.answer_e = tf.placeholder(tf.int32, (None,))
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="dropout", shape=())
        with tf.variable_scope("qa",
                               initializer=tf.uniform_unit_scaling_initializer(1.0,)):
            self.embeddings_init()
            self.system_init()
            self.loss_init()

            self.global_step = tf.Variable(0, trainable=False)
            self.starter_learning_rate = tf.placeholder(tf.float32, name='start_lr')
            learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                       1000, 0.96, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            gradients = self.optimizer.compute_gradients(self.final_loss)
            capped_gvs = [(tf.clip_by_value(grad, -cfg.clip_by_val, cfg.clip_by_val), var) for grad, var in gradients]
            grad = [x[0] for x in gradients]
            self.grad_norm = tf.global_norm(grad)
            tf.summary.scalar('grad_norm', self.grad_norm)
            self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()
            
    def remove_punc(self, text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def remove_articles(self, text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # preprocess the answer for clean-up and input to the embeddings
    def preprocess_answer(self, s):
        return ' '.join(self.remove_articles(self.remove_punc(s.lower())))


    # F1 metric as a performance metric
    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.preprocess_answer(prediction).split()
        ground_truth_tokens = self.preprocess_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    
    # Exact Match as a performance metric
    def EM_score(self, prediction, ground_truth):
        return (self.preprocess_answer(prediction) == self.preprocess_answer(ground_truth))


    def system_init(self):
        H_r = self.encoder.encode(self.context,
                                    self.context_m, self.question,
                                    self.question_m, self.embedding, self.keep_prob)
        self.s_score, self.e_score = self.decoder.decode(H_r, self.context_m, self.keep_prob)

    def loss_init(self):
        with vs.variable_scope("loss"):
            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.s_score, labels=self.answer_s)

            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.e_score, labels=self.answer_e
            )
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        self.final_loss = tf.reduce_mean(loss_e + loss_s) + reg_term
        tf.summary.scalar('final_loss', self.final_loss)

    def embeddings_init(self):
        logging.info('embed size: {} for path {}'.format(cfg.embed_size, self.embed_path))
        self.embedding = np.load(self.embed_path)['glove']
        self.embedding = tf.Variable(self.embedding, dtype=tf.float32, trainable=False)

    # prepare the input and output dict for a tensorflow Session
    # process to be fed into training
    def optimize(self, session, context, question, answer, lr):
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]
        answer_start = [x[0] for x in answer]
        answer_end = [x[1] for x in answer]

        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m: question_masks,
                      self.answer_s: answer_start,
                      self.answer_e: answer_end,
                      self.starter_learning_rate: lr,
                      self.keep_prob: cfg.keep_prob}

        output_feed = [self.merged, self.train_op, self.final_loss, self.grad_norm]
        outputs = session.run(output_feed, input_feed)
        return outputs

    # prepare the input and output dict for a tensorflow Session
    # process for inferencing
    def decode(self, session, context, question):
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]

        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m: question_masks,
                      self.keep_prob: 1.}

        output_feed = [self.s_score, self.e_score]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, context, question,train = True):
        yp, yp2 = self.decode(session, context, question)        
        if train:
            a_s = np.argmax(yp, axis=1)
            a_e = np.argmax(yp2, axis=1)
        else:
            a_s, a_e = proc_max_prob(yp, yp2)
        return a_s, a_e


    def evaluate_answer(self, session, dataset, answers, rev_vocab,
                        set_name='val', training=False, log=False,
                        sample=(100, 100), sendin=None, ensemble=False):

        if not isinstance(rev_vocab, np.ndarray):
            rev_vocab = np.array(rev_vocab)

        if not isinstance(sample, tuple):
            sample = (sample, sample)

        input_batch_size = 100

        if training:
            train_context = dataset['train_context'][:sample[0]]
            train_question = dataset['train_question'][:sample[0]]
            train_answer = answers['raw_train_answer'][:sample[0]]
            train_len = len(train_context)

            if sendin and len(sendin) > 2:
                train_a_s, train_a_e = sendin[0:2]
            else:
                train_a_e = np.array([], dtype=np.int32)
                train_a_s = np.array([], dtype=np.int32)

                for i in tqdm(range(train_len // input_batch_size), desc='training set'):
                    train_as, train_ae = self.answer(session,
                                                     train_context[i * input_batch_size:(i + 1) * input_batch_size],
                                                     train_question[i * input_batch_size:(i + 1) * input_batch_size])
                    train_a_e = train_a_e+ np.random.randint(20)
                    train_a_s = np.concatenate((train_a_s, train_as), axis=0)
                    train_a_e = np.concatenate((train_a_e, train_ae), axis=0)
                    

            tf1 = 0.
            tem = 0.
            for i, con in enumerate(train_context):
                sys.stdout.write('>>> %d / %d \r' % (i, train_len))
                sys.stdout.flush()
                
                if train_a_e[i]>= len(con[0]):
                    train_a_e[i] = len(con[0])-1
                
                prediction_ids = con[0][train_a_s[i]: train_a_e[i] + 1]
                prediction = rev_vocab[prediction_ids]
                prediction = ' '.join(prediction)
                tf1 += self.f1_score(prediction, train_answer[i])

#            if log:
#                logging.info("Training set ==> F1: {}".
#                             format(tf1 / train_len))

        # initializing the f1 and EM scores 
        # to be aggregated across the samples
        f1 = 0.0
        em = 0.0
        val_context = dataset[set_name + '_context'][:sample[1]]
        val_question = dataset[set_name + '_question'][:sample[1]]
        val_answer = answers['raw_val_answer'][:sample[1]]

        val_len = len(val_context)

        if sendin and len(sendin) > 2:
            val_a_s, val_a_e = sendin[-2:]
        elif sendin:
            val_a_s, val_a_e = sendin
        else:
            val_a_s = np.array([], dtype=np.int32)
            val_a_e = np.array([], dtype=np.int32)
            for i in tqdm(range(val_len // input_batch_size), desc='validation   '):
                a_s, a_e = self.answer(session, val_context[i * input_batch_size:(i + 1) * input_batch_size],
                                       val_question[i * input_batch_size:(i + 1) * input_batch_size])
                a_e = a_e+ np.random.randint(20)                       
                val_a_s = np.concatenate((val_a_s, a_s), axis=0)
                val_a_e = np.concatenate((val_a_e, a_e), axis=0)

        for i, con in enumerate(val_context):
            sys.stdout.write('>>> %d / %d \r' % (i, val_len))
            sys.stdout.flush()
            if i==1800:
                break
            if val_a_e[i]>= len(con[0]):
                val_a_e[i] = len(con[0])-1
            
            prediction_ids = con[0][val_a_s[i]: val_a_e[i] + 1]
            prediction = rev_vocab[prediction_ids]
            prediction = ' '.join(prediction)
            f1 += self.f1_score(prediction, val_answer[i])

#        if log:
#            logging.info("Validation   ==> F1: {}".
#                         format(f1 / val_len))
   

        if ensemble and training:
            return train_a_s, train_a_e, val_a_s, val_a_e
        elif ensemble:
            return val_a_s, val_a_e
        else:
            return tf1 / train_len, tem / train_len, f1 / val_len, em / val_len

    # Training routine
    def train(self, lr, session, dataset, answers, train_dir, debug_num=0, raw_answers=None,
              rev_vocab=None):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        train_context = np.array(dataset['train_context'])
        train_question = np.array(dataset['train_question'])
        train_answer = np.array(answers['train_answer'])

        if cfg.valohai:
            print_every = cfg.print_every // 2
        else:
            print_every = cfg.print_every

        if debug_num:
            assert isinstance(debug_num, int), 'the debug number should be a integer'
            assert debug_num < len(train_answer), 'check debug number!'
            train_answer = train_answer[0:debug_num]
            train_context = train_context[0:debug_num]
            train_question = train_question[0:debug_num]
            print_every = 5

        num_example = len(train_answer)
        logging.info('num example is {}'.format(num_example))
        shuffle_list = np.arange(num_example)
        
        # record training stats
        self.epochs = cfg.epochs
        self.losses = []
        self.norms = []
        self.train_eval = []
        self.val_eval = []
        self.iters = 0
        save_path = pjoin(train_dir, 'weights')
        self.train_writer = tf.summary.FileWriter(cfg.summary_dir + str(lr),
                                                  session.graph)

        batch_size = cfg.batch_size
        batch_num = int(num_example / batch_size)
        total_iterations = self.epochs * batch_num
        tic = time.time()

        for ep in xrange(self.epochs):
            # TODO: add random shuffle.
            np.random.shuffle(shuffle_list)
            train_context = train_context[shuffle_list]
            train_question = train_question[shuffle_list]
            train_answer = train_answer[shuffle_list]

            logging.info('training epoch ---- {}/{} -----'.format(ep + 1, self.epochs))
            ep_loss = 0.
            for it in xrange(batch_num):
                if not cfg.valohai:
                    sys.stdout.write('> %d / %d \r' % (self.iters % print_every, print_every))
                    sys.stdout.flush()
                context = train_context[it * batch_size: (it + 1) * batch_size]
                question = train_question[it * batch_size: (it + 1) * batch_size]
                answer = train_answer[it * batch_size: (it + 1) * batch_size]
                

                outputs = self.optimize(session, context, question, answer, lr)
                self.train_writer.add_summary(outputs[0], self.iters)
                loss, grad_norm = outputs[2:]

                ep_loss += loss
                self.losses.append(loss)
                self.norms.append(grad_norm)
                self.iters += 1

                if self.iters % print_every == 0:
                    toc = time.time()
                    logging.info('iters: {}/{} loss: {} norm: {}. time: {} secs'.format(
                        self.iters, total_iterations, loss, grad_norm, toc - tic))

                    tf1, tem, f1, em = self.evaluate_answer(session, dataset, raw_answers, rev_vocab,
                                                            training=True, log=True, sample=cfg.sample)
                    
                    self.train_eval.append((tf1, tem))
                    self.val_eval.append((f1, em))
                    tic = time.time()

                if self.iters % cfg.save_every == 0:
                    self.saver.save(session, save_path, global_step=self.iters)
                    self.evaluate_answer(session, dataset, raw_answers, rev_vocab,
                                         training=True, log=True, sample=4000)
            if cfg.save_every_epoch:
                self.saver.save(session, save_path, global_step=self.iters)

            logging.info('average loss of epoch {}/{} is {}'.format(ep + 1, self.epochs, ep_loss / batch_num))

            data_dict = {'losses': self.losses, 'norms': self.norms,
                         'train_eval': self.train_eval, 'val_eval': self.val_eval}
            c_time = time.strftime('%Y%m%d_%H%M', time.localtime())
            data_save_path = pjoin(cfg.cache_dir, str(self.iters) + 'iters' + c_time + '.npz')
            np.savez(data_save_path, data_dict)
        
