#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:14:36 2018

@author: snehal
"""

import pickle
from config import Config as cfg
import os

#This file add new sentiment based QA pairs to the dataset

def main():
    data = pickle.load( open(os.path.join(cfg.DATA_DIR, "Amazon_data.pkl"), "rb" ) )


    features_appliances=['oz','warranty', 'usage', 'capacity', 'durable', #4000\ 
                        'color', 'colour', 'fabric', 'quality', 'print', 'manual', #23440\ 
                        'reusable','removable', 'safe','child-resistant', 'protect',#26205
                        'measure', 'elegant', 'soft', 'easy', 'size', 'strength',
                        'strong', 'grip'] #56455

    ques_feat_appl=['Is the capacity of the product enough?',\
                            'Does the product give good warranty?',\
                            'Is the product useful?',\
                            'Is the capacity of the product enough?',\
                            'Is the product durable?',\
                            'Is the color of the product vibrant?',\
                            'Is the color of the product vibrant?',\
                            'How is the fabric of the product?',\
                            'What can you say about its quality?',\
                            'Is the print neat?',\
                            'Is there a manual with the product?',\
                            'Is this product reusable?',\
                            'Is the product removable?',\
                            'Is it safe using this product?',\
                            'Is it safe for children?',\
                            'Does this product provide a protection?',
                            'Is the product spacious?',
                            'Is it elegant?',
                            'Does the product feel soft',
                            'Is the product easy to use',
                            'Is the size appropriate?',
                            'How is the strength of the product',
                            'Is the product strong?',
                            'Does it have a good grip']
    ans_pos_feat_appl=['Yes it is enough',\
                            'Yes the product gives good warranty',\
                            'Yes the product is useful',\
                            'Yes the capacity of the product is enough',\
                            'Yes the product is durable',\
                            'Yes the color of the product is vibrant',\
                            'Yes the color of the product is vibrant',\
                            'The fabric of the product is good',\
                            'The quality of the product is good',\
                            'The print is very neat',\
                            'Yes there is a manual with the product',\
                            'Yes the product is reusable?',\
                            'Yes the product is removable?',\
                            'Yes it is safe using this product?',\
                            'Yes it safe for children?',\
                            'Yes this product provides a protection?',
                            'Yes the product is spacious?',
                            'Yes the product is elegant?',
                            'Yes the product feels soft',
                            'Yes the product is easy to use',
                            'The size is appropriate?',
                            'The strength of the product is appropriate',
                            'Yes the product is strong?',
                            'Yes the product has a good grip']

    ans_neg_feat_appl=['The capacity is not enough',\
                            'The product does not give good warranty',\
                            'The product is not useful',\
                            'The capacity of the product is not enough',\
                            'The product is not durable',\
                            'The color of the product is not vibrant',\
                            'The color of the product is not vibrant',\
                            'The fabric of the product is not good',\
                            'The quality of the product is not good',\
                            'The print is not very neat',\
                            'There is not a manual with the product',\
                            'The product is not reusable?',\
                            'The product is not removable?',\
                            'It is not safe using this product?',\
                            'It is not safe for children?',\
                            'This product does not provide a protection?',
                            'The product is not spacious?',
                            'The product is not elegant?',
                            'The product does not feel soft',
                            'The product is not easy to use',
                            'The size is not appropriate?',
                            'The strength of the product is not appropriate',
                            'The product is not strong?',
                            'The product does not have a good grip']

    count=0
    count_1=0
    category_name=list(data.items())
    features=[]
    new_context=[]
    unique_keywords=[]
    for i in range(len(data)):
        asins_dict=category_name[i][1]
        asin_number=list(asins_dict.items())
        for j in range(len(asin_number)):
            QAs=data[category_name[i][0]][asin_number[j][0]]['QA']
            for k in range(len(QAs)):
                count=count+1
                QAs[k].append(0)
            context=data[category_name[i][0]][asin_number[j][0]]['context']
            keywords=data[category_name[i][0]][asin_number[j][0]]['keywords']
            looked_keywords=[]
            for l in range(len(keywords)):
                #for m in range(len(keywords[l])):
                key_phrases=keywords[l][0].split()
                pos=float(keywords[l][1])
                neg=float(keywords[l][2])
                for word in key_phrases:
                    if word in features_appliances:
                        
                        if word not in looked_keywords:
                            looked_keywords.append(word)
                            
                            
                            question1=ques_feat_appl[features_appliances.index(word)]
                            if pos>neg:
                                
                                ans1=ans_pos_feat_appl[features_appliances.index(word)]
                            else:
                                ans1=ans_neg_feat_appl[features_appliances.index(word)]
                            
                            QAs.append([question1,ans1,1])
                            context=context+ '\n' +ans1

                        
                        
                        
            question2='What is the price of the product'
            split_context=context.split('.')
            for m in range(len(split_context)):
                if split_context[m].find('The price is') !=-1:
                    ans2=split_context[m].split()
                    if len(ans2)!=0:
                        ans2=ans2[-1]
                        ans2_1=ans2+'.'+split_context[m+1]
                        if ans2[0]!='$':
                            count_1+=1
                            ans2_1=' The price is not available'
                            context=context.replace('The price is', 'The price is not available')
                            new_context.append(context)
                            
                    else:
                        count_1+=1
                        context=context.replace('The price is', 'The price is not available')
                        ans2_1=' The price is not available'
                        new_context.append(context)



            QAs.append([question2,ans2_1,1])

            data[category_name[i][0]][asin_number[j][0]]['QA']=QAs
            data[category_name[i][0]][asin_number[j][0]]['context']=context

    pickle.dump( data, open(os.path.join(cfg.DATA_DIR, "amz_new_qa_pairs.pkl"), "wb" ) )

if(__name__=="__main__"):
    main()
