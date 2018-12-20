# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:13:05 2018
Change from answers to reviews
yet to write the complete data
@author: Nadha
"""

import gzip
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize
import pandas as pd
from collections import defaultdict
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
import os
from collections import defaultdict
import pickle
from argparse import ArgumentParser
from glob import glob
from config import Config as cfg

class dataProc:
    def __init__(self, feature_based_keywords, threshold):
        self.threshold = threshold
        self.feature_based_keywords = feature_based_keywords
        self.r = Rake()
        self.sid = SentimentIntensityAnalyzer()
        self.word_index = {}
        
    def parse(self, path):
      g = gzip.open(path, 'rb')
      for l in g:
        yield eval(l)
    
    def getDF(self, path):
      i = 0
      df = {}
      for d in self.parse(path):
        df[i] = d
        i += 1
      return pd.DataFrame.from_dict(df, orient='index')

    # function to get sort QA pairs and combine all the answers according to asins
    def process_data(self, path, category_list):
        filename = os.path.join(path, '*.json.gz')
        it = glob(filename)
        #print(it)
        l = []  
       
        combined_answers_category = defaultdict(dict)
        question_category = defaultdict(dict) 
        for cat in category_list:
            ele = os.path.join(path,cat)
            df = self.getDF(ele)
            l.append(list(df.iloc[:,1]))
    
            # storing the answers
            d = list(df['answer'])
            questions = list(df['question'])
            
            asin = list(df['asin'])
            # storing the sentiment scores
            qa = list(zip(asin,questions,d))
            
            qa_dict = defaultdict(list)
            for i, j, k in qa:
                qa_dict[i].append([j,k]) 
            question_category[ele] = qa_dict
            
            
            res = defaultdict(list) 
            z = list(zip(asin, d))
            
            for i, j in z:
                res[i].append(j)
            
                
            dt = {k:' '.join(v) for k, v in res.items()}
            combined_answers_category[ele] = dt
            
        return combined_answers_category, question_category

    
   # function to build a sentiment histgram for keywords from review
    def build_sentiment_histogram(self, review_keyword_list, review_sentiment_list):
       pos_dict = defaultdict(int)
       neg_dict = defaultdict(int)
       neu_dict = defaultdict(int)
       keyword_dict = {}
       for i in range(len(review_keyword_list)):
           for j in review_keyword_list[i]:
               if j not in keyword_dict:
                   keyword_dict[j] ={}
                   keyword_dict[j]['pos']=0
                   keyword_dict[j]['neg'] =0
                   #ekeyword_dict[j]['neu']=0
               keyword_dict[j][review_sentiment_list[i]] +=1
       return keyword_dict

    # function to combine scraped data and QA pairs from dataset according to their asins
    # to get context, question and answers
    def concatnate_desc_review(self,path, review_path,category_list,combined_answers_category,question_category,save_path):
        data_dict = defaultdict(dict)
        features=['oz','warranty', 'usage', 'capacity', 'durable', #4000\
                    'color', 'colour', 'fabric', 'quality', 'print', 'manual', #23440\
                    'reusable','removable', 'safe','child-resistant', 'protect',#26205
                    'measure', 'elegant', 'soft', 'easy', 'size', 'strength',
                    'strong', 'grip']
        for i in category_list:
            category_data = defaultdict(dict)
            answer_path = os.path.join(path,i)
            answer_dict = combined_answers_category[answer_path]
            question_dict = question_category[answer_path]
            review_category_path = os.path.join(review_path,i)
            name_list = os.listdir(review_category_path)
            
            for j in name_list:
                with open(os.path.join(review_category_path,j),'rb') as handle:
                    try:
                        b = pickle.load(handle)
                        asin = j.split('.')[0]
                    
                        context = 'Product name is ' + b['name'] + ". "
                        if not b['proddesc']:
                            context = context + b['proddesc'][0]+". "
                        reviews = ' '.join([k['review_text'] for k in b['reviews']])
                        
                        sentiments = []
                        keywords = []
                        for k in range(len(b['reviews'])):
                            ele1 = b['reviews'][k]['review_text']
                            sentiments.append(self.sid.polarity_scores(ele1))
                            self.r.extract_keywords_from_text(ele1)
                            keywords.append( self.r.get_ranked_phrases())
                            
                            
                        s = [None]*len(b['reviews'])
                        for k, ele1 in enumerate(sentiments):
                            if ele1['compound'] >self.threshold:
                                s[k] = 'pos'
                            else:
                                s[k]= 'neg'
                        
                        review_senti_dict = self.build_sentiment_histogram(keywords,s)
                        
                        
                        context = context + reviews+'. '
                        context = context + answer_dict[asin]+'. '
                        context = context + ' The price is '+ b['price'] + '. '
                        
                        updated_keywords = []
                        for k,v in review_senti_dict.items():
                            
                            if v['pos']>1 or v['neg']>1:
                                pos = str((100*v['pos'])/(v['pos']+v['neg']))
                                neg = str((100*v['neg'])/(v['pos']+v['neg']))
                                updated_keywords.append((k,pos,neg))
                                context = context + k + " is liked by: "+ pos[:4]+ " and disliked by: "+ neg[:4] +'. ' 
                            else:
                                for z in features:
                                    if z in k.split():
                                        pos = str((100*v['pos'])/(v['pos']+v['neg']))
                                        neg = str((100*v['neg'])/(v['pos']+v['neg']))
                                        updated_keywords.append((z,pos,neg))
                                        context = context + z + " is liked by: "+ pos[:4]+ " and disliked by: "+ neg[:4] +'. ' 
                        
                        
                        category_data[asin] = {}
                        category_data[asin]['context'] = context
                        category_data[asin]['QA'] = question_dict[asin]
                        #category_data[asin]['sentiment']= s
                        category_data[asin]['keywords'] = updated_keywords
                    except:
                        #print(e)
                        pass
                
            data_dict[i] = category_data
                      
        pickle.dump(data_dict, open(save_path, "wb" ))
    
if __name__ == "__main__":
    feature_based_keywords = ['salient', 'features', 'feature', 'important']
    parser = ArgumentParser()
    parser.add_argument("Category_data",
                        help="Path to Categorical data", type =str)

    parser.add_argument("Review_path", help="Path to review data", type =str)
                    

    args = parser.parse_args()
    path = args.Category_data
    review_path = args.Review_path
    if not os.path.exists(cfg.DATA_DIR):
        os.makedirs(cfg.DATA_DIR)
    save_path = os.path.join(cfg.DATA_DIR, "Amazon_data.pkl")
    # ad-hoc threshold which will be changed in the future
    threshold = 0
    
    # instantiate the class 
    data = dataProc(feature_based_keywords, threshold)
    
    category_list = ['qa_Appliances.json.gz','qa_Arts_Crafts_and_Sewing.json.gz',
                     'qa_Baby.json.gz','qa_Automotive.json.gz','qa_Cell_Phones_and_Accessories.json.gz',
                    'qa_Clothing_Shoes_and_Jewelry.json.gz', 'qa_Sports_and_Outdoors.json.gz',
                    'qa_Toys_and_Games.json.gz',
                            'qa_Home_and_Kitchen.json.gz', 'qa_Patio_Lawn_and_Garden.json.gz']
                            

    combined_answers_category, qa_dict = data.process_data(path,category_list)
    
    data.concatnate_desc_review(path, review_path, category_list, combined_answers_category, qa_dict, save_path)
    
    