import nltk
import numpy as np
import os
import re
import pickle
import argparse
from os.path import join as pjoin
from tensorflow.python.platform import gfile
from tqdm import tqdm
from config import Config as cfg

class DataLoader:
    def __init__(self):
        self._PAD = b"<pad>"
        self._SOS = b"<sos>"
        self._UNK = b"<unk>"

        self.PAD_ID = 0
        self.SOS_ID = 1
        self.UNK_ID = 2
        self._START_VOCAB = [self._PAD, self._SOS, self._UNK]
        self.glove_dir = pjoin("data", "embedding")
        self.glove_dim = 100
    def normalize(self, dat):
        return list(map(lambda tok: map(int, tok.split()), dat))

    def tokenize(self, sequence):
        tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
        return list(map(lambda x: x.encode('utf-8'), tokens))
    
    def ans_search_span_single(self, context, ans):
        ans_start = context.find(ans)
        return ans_start, ans_start + len(ans)
    
    def process_amazon_data(self, pickle_file, parsed_pickle_name):
        final_dict = {}
        d_2 = open(pickle_file, 'rb')
        b = pickle.load(d_2)
        context = []
        question = []
        answer = []
        answer_start = []
        answer_end = []
        q = 0
        keys = list(b.keys())
        for k in keys:
            for k1 in list(b[k].keys()):
                l = len(b[k][k1]['QA'])
                for i in range(l):
                    context.append(b[k][k1]['context'])
                    question.append(b[k][k1]['QA'][i][0])
                    answer.append(b[k][k1]['QA'][i][1])
                    ans_st, ans_end = self.ans_search_span_single(b[k][k1]['context'],
                                                             b[k][k1]['QA'][i][1])
                    q += 1
                    answer_start.append(ans_st)
                    answer_end.append(ans_end)
        
            
        final_dict['context'] = context
        final_dict['question'] = question
        final_dict['answer_start'] = answer_start
        final_dict['answer_end'] = answer_end
        final_dict['answer'] = answer
        pickle.dump(final_dict, open(parsed_pickle_name, 'wb'), protocol=2)
        return final_dict
   


    def token_sent(self, sentence):
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(re.split(" ", space_separated_fragment))
        return [w for w in words if w]


    def initialize_vocabulary(self, vocabulary_path):
        # map vocab to word embeddings
        if gfile.Exists(vocabulary_path):
            rev_vocab = []
            with gfile.GFile(vocabulary_path, mode="r") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip('\n') for line in rev_vocab]   
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return vocab, rev_vocab
        else:
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)


    # creates an numpy representation of the glove embeddings for each word
    # stores it in a saved file 
    def process_glove(self, vocab_list, save_path, size=4e5):
        if not gfile.Exists(save_path + ".npz"):
            glove_path = os.path.join(self.glove_dir, "glove.6B.{}d.txt".format(self.glove_dim))
            glove = np.zeros((len(vocab_list), self.glove_dim))
            found = 0
            with open(glove_path, 'r') as fh:
                for line in tqdm(fh, total=size):
                    array = line.lstrip().rstrip().split(" ")
                    word = array[0]
                    vector = list(map(float, array[1:]))
                    if word in vocab_list:
                        idx = vocab_list.index(word)
                        glove[idx, :] = vector
                        found += 1
                    if word.capitalize() in vocab_list:
                        idx = vocab_list.index(word.capitalize())
                        glove[idx, :] = vector
                        found += 1
                    if word.upper() in vocab_list:
                        idx = vocab_list.index(word.upper())
                        glove[idx, :] = vector
                        found += 1

            np.savez_compressed(save_path, glove=glove)


    def create_vocabulary(self, vocabulary_path, data_paths, tokenizer=None):
        if not gfile.Exists(vocabulary_path):
            print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
            vocab = {}
            for path in data_paths:
                with open(path, mode="rb") as f:
                    counter = 0
                    for line in f:
                        counter += 1
                        if counter % 100000 == 0:
                            print("processing line %d" % counter)
                        tokens = tokenizer(line) if tokenizer else self.token_sent(line)
                        for w in tokens:
                            if w in vocab:
                                vocab[w] += 1
                            else:
                                vocab[w] = 1
            vocab_list = self._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            print("Vocabulary size: %d" % len(vocab_list))
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


    def sentence_to_token_ids(self, sentence, vocabulary, tokenizer=None):
        if tokenizer:
            words = tokenizer(sentence)
        else:
            words = self.token_sent(sentence)
        return [vocabulary.get(w, self.UNK_ID) for w in words]


    def data_to_token_ids(self, data_path, target_path, vocabulary_path,
                          tokenizer=None):
        if not gfile.Exists(target_path):
            print("Tokenizing data in %s" % data_path)
            vocab, _ = self.initialize_vocabulary(vocabulary_path)
            with gfile.GFile(data_path, mode="r") as data_file:
                with gfile.GFile(target_path, mode="w") as tokens_file:
                    counter = 0
                    for line in data_file:
                        counter += 1
                        if counter % 5000 == 0:
                            print("tokenizing line %d" % counter)
                        token_ids = self.sentence_to_token_ids(line, vocab, tokenizer)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

    def word_parse_single(self, para_split, text_split):
        cursor = 0
        found = []
    
        for i in range(len(para_split)):
            if para_split[i] == text_split[cursor]:
                cursor += 1
                if cursor == len(text_split):
                    found.append((i - len(text_split), i))
                    break
            else:
                cursor = 0
       
        return found
       
    def _pad_sequences(self, sequences, pad_tok, max_length):
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded +=  [seq_]
            sequence_length += [min(len(seq), max_length)]

        return np.array(sequence_padded), np.array(sequence_length)


    def pad_sequences(self, sequences, pad_tok):
        max_length = max([len(x) for x in sequences])
        sequence_padded, sequence_length = self._pad_sequences(sequences, 
                                                pad_tok, max_length)

        return sequence_padded, sequence_length


    def minibatches(self, data, minibatch_size):
        question_batch, context_batch, answer_batch = [], [], []

        for (q, c, a) in data:
            if len(question_batch) == minibatch_size:
                yield question_batch, context_batch, answer_batch
                question_batch, context_batch, answer_batch = [], [], []
            
            question_batch.append(q)
            context_batch.append(c)
            answer_batch.append(a)

        if len(question_batch) != 0:
            yield question_batch, context_batch, answer_batch

    # load glove vectors
    def get_glove_vectors(self, filename):
        return np.load(filename)["glove"]


    # find cases where answers are not available     
    def find_special(self, context, ans_span):
        special = []
        print ('special')
        for i in range(len(context)):
            text = context[i][ans_span[i][0]:ans_span[i][1]]
            para_split = context[i].split()
            text_split = text.split()
            if len(text_split) == 0:
                special.append(i)
            else:
                f = self.word_parse_single(para_split, text_split)
                if len(f) == 0:    
                    special.append(i)
        return special
    
    def word_parse(self, context, ans_span):
        found = []
        exp = []
        special = self.find_special(context, ans_span)
        for i in range(len(context)):
            if i in special:
                found.append([(tuple([0, 0]))])
            else:
                text = context[i][ans_span[i][0]:ans_span[i][1]]
                para_split = context[i].split()
                text_split = text.split()
                f = self.word_parse_single(para_split, text_split)
                if len(f) != 0:
                    if f[0][1] > len(context[i].split()):
                        
                        exp.append(i)
                    found.append(f)
        return found, exp 

    def load_parsed_amazon_data(self, pickle_file, num_start, num_end):
        f1 = open(pickle_file, 'rb')
        data_dict = pickle.load(f1)
#        k = list(data_dict.keys())
#        m = np.random.permutation(len(data_dict[k[0]]))
        context = np.array(data_dict['context'])
        question = np.array(data_dict['question'])
        ans_st = np.array(data_dict['answer_start'])
        ans_end = np.array(data_dict['answer_end'])
        ans_span = list(zip(ans_st, ans_end))
        ans_st = list(map(lambda x: x[0], ans_span))
        ans_end = list(map(lambda x: x[1], ans_span))
        return context[num_start:num_end], question[num_start:num_end], ans_st[num_start:num_end], ans_end[num_start:num_end]
        
    
    def write_parsed_data(self, tier, context, question, ans_st, ans_end):
        with open(os.path.join(cfg.DATA_DIR,tier +'.context'), 'wb') as context_file,  \
         open(os.path.join(cfg.DATA_DIR, tier +'.question'), 'wb') as question_file, \
         open(os.path.join(cfg.DATA_DIR,tier +'.span'), 'w') as span_file:
             for i in range(len(context)):
                 context_tokens = self.tokenize(context[i])
                 question_tokens = self.tokenize(question[i])
                 context_file.write(b' '.join(context_tokens) + b'\n')
                 question_file.write(b' '.join(question_tokens) + b'\n')
                 span_file.write(' '.join([str(ans_st[i]), str(ans_end[i])]) + '\n')
                 
                 
if __name__ == "__main__":
    data = DataLoader()
    vocab_path = pjoin(cfg.DATA_DIR, "vocab.dat")

    train_path = pjoin(cfg.DATA_DIR, "train")
    valid_path = pjoin(cfg.DATA_DIR, "val")

    d2 = data.process_amazon_data(pjoin(cfg.DATA_DIR, 'test_cat.pkl'), pjoin(cfg.DATA_DIR, 'amazon_parse_test.pkl'))

    d1 = data.process_amazon_data(pjoin(cfg.DATA_DIR, 'train_cat_minimal.pkl'), pjoin(cfg.DATA_DIR, 'amazon_parse_train.pkl'))
    print('Training')
    if not os.path.exists(pjoin(cfg.DATA_DIR, 'amazon_parse_train.pkl')):
        c, q, a_s, a_e = data.load_parsed_amazon_data(pjoin(cfg.DATA_DIR, 'amazon_parse_train.pkl'), 0, 9501)
        print('Writing train file')
        data.write_parsed_data('train', c, q, a_s, a_e)
    print('Test')
    if not os.path.exists(pjoin(cfg.DATA_DIR, 'amazon_parse_test.pkl')): 
        c1, q1, a_s1, a_e1 = data.load_parsed_amazon_data(pjoin(cfg.DATA_DIR, 'amazon_parse_test.pkl'),0, int(9501*0.2))
        print('Writing test file')
        data.write_parsed_data('test', c1, q1, a_s1, a_e1)
    print('Val')
    if not os.path.exists(pjoin(cfg.DATA_DIR, 'amazon_parse_test.pkl')):
        c2, q2, a_s2, a_e2 = data.load_parsed_amazon_data(pjoin(cfg.DATA_DIR, 'amazon_parse_test.pkl'),int(9501*0.2) + 1, int(9501*0.4))
        print('Writing val file')
        data.write_parsed_data('val', c2, q2, a_s2, a_e2)

    data.create_vocabulary(vocab_path, [pjoin(cfg.DATA_DIR, "train.context"), pjoin(cfg.DATA_DIR, "train.question"), pjoin(cfg.DATA_DIR, "val.context"), pjoin(cfg.DATA_DIR, "val.question")])
    x_train_dis_path = train_path + ".ids.context"
    y_train_ids_path = train_path + ".ids.question"
    print('Tokenizing the train file')
    data.data_to_token_ids(train_path + ".context", x_train_dis_path, vocab_path)
    data.data_to_token_ids(train_path + ".question", y_train_ids_path, vocab_path)

    x_dis_path = valid_path + ".ids.context"
    y_ids_path = valid_path + ".ids.question"
    print('Tokenizing the val file')
    data.data_to_token_ids(valid_path + ".context", x_dis_path, vocab_path)
    data.data_to_token_ids(valid_path + ".question", y_ids_path, vocab_path)
    