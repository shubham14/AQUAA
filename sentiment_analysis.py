# -*- coding: utf-8 

import pickle
from os.path import join as pjoin
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

def sentiment_dict(pickle_file):
    '''
    Parses the amazon pickle file to extract sentiment based analysis
    '''
    file_path = pjoin('amazon', pickle_file)
    ama = pickle.load(open(file_path, 'rb'))
    keys = list(ama.keys())
    keywords_dict = {}
    for key in keys:
        keywords = []
        for asin in list(ama[key].keys()):
            l = ama[key][asin]['keywords']
            for ele in l:
                # assigning the negative and positive sentiments
                if ele[1] > ele[2]:
                    keywords.append((ele[0], 'pos'))
                else:
                    keywords.append((ele[0], 'neg'))
        keywords_dict[key] = keywords
        keys = list(keywords_dict.keys())
    
        # storing the number of positve and negative sentiments 
        # for each category
        length = defaultdict(dict)
        sentiment_keywords = defaultdict(dict)
        for key in keys:
            pos = []
            neg = []
            v = dict(Counter(list(keywords_dict[key])))
            for k in list(v.keys()):
                if k[1] == 'pos':
                    pos.append(k[0])
                else:
                    neg.append(k[0])
            sentiment_keywords[key]['pos'] = pos
            sentiment_keywords[key]['neg'] = neg
            pos_ratio = float(len(pos)) / (len(pos) + len(neg))
            neg_ratio = float(len(neg)) / (len(pos) + len(neg))
            length[key]['pos'] = pos_ratio
            length[key]['neg'] = neg_ratio
            
    return sentiment_keywords, length


def sent_hist(length):
    '''
    Plots a category-wise sentiment histogram from the 
    mined opinions
    '''
    pos = []; neg = []
    leg = ['positive', 'negative']
    for k in list(length.keys()):
        pos.append(length[k]['pos'])
        neg.append(length[k]['neg'])
        
    # plot for x-axis
    k = np.array([i for i in range(1, len(list(length.keys())) + 1)])
   
    ax = plt.subplot(111)
    ax.bar(np.array(k)-0.3, pos, width=0.3, color='b',align='center')
    ax.bar(np.array(k), neg, width=0.3, color='g',align='center')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Ratio of sentiment reviews')
    plt.xticks([1,2,3,4,5,6,7,8,9,10])
    plt.legend(leg, loc=0)
    
    plt.savefig('Sentiment_hist.jpg', dpi=500)
    
if __name__ == "__main__":
    
    sentiment_keywords, length = sentiment_dict('Amazon_data_minimal.pkl')
    sent_hist(length)