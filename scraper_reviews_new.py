#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:05:38 2018

@author: snehal
"""
#from bs4 import BeautifulSoup
import json 
import gzip 
from time import sleep
from lxml import html  
import json
import requests
import json,re
from dateutil import parser as dateparser
import pandas as pd
from fake_useragent import UserAgent
import random
import os
from lxml import html
from json import dump,loads
from requests import get
import json
from re import sub
from dateutil import parser as dateparser
from time import sleep
import pickle

ua = UserAgent()
##DONE TLL COUNT 2252
##LENGTH OF EXTRACTED_DATA=750
##PROD_ID=B0089FUFA8
#Released for the first time with a new cut by director Martin Huberty, Deserter is the gritty true story of an idealistic young English gentleman Simon Murray (Paul Fox) driven to join the French Foreign Legion after a failed romance. Simon s romantic illusions are soon shattered when he encounters his fellow recruits, including Dupont (Tom Hardy.) They have all volunteered to escape a past far less innocent than Simon s. A sadistic training routine forges the recruits together. As fully-fledged Legionnaires they set out to do battle against the insurgents facing deadly firefights among the villages, hills and deserts of North Africa. But when French President De Gaulle grants Algerian independence, the Legion sides with the Pied Noir French Nationalists against mother France. Loyalties are torn apart. Comrade is turned against comrade. Simon and Dupont must face each other across a moral divide, their conscience dictating their actions; blurring the distinction between hero and deserter.
# def get_proxies():
#     """Retrieves a list of proxies"""

#     proxies = set()

#     # eventually put this somewhere else
#     proxy_sources = [
#         'https://free-proxy-list.net/anonymous-proxy.html', 
#         'https://www.us-proxy.org/', 
#         'https://www.sslproxies.org/', 
#         'https://www.socks-proxy.net/'
#     ]

#     # count times this is executed. stop at 10 attempts.
#     attempt = 0
#     while not len(proxies) > 0:
#         for source in proxy_sources:
#             res = requests.get(source, headers={
#                 'User-Agent':ua.random
#                 })

#             if res.status_code != 200:
#                 print("connection error " + str(res.status_code) \
#                     + " source " + source)
#             else:
#                 soup = BeautifulSoup(res.content, 'html.parser')
#                 tab = soup.find("table", {"id":"proxylisttable"})
#                 for cell in tab.find_all('td'):
#                     if cell.string != None and re.match('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', cell.string) != None: 
#                         proxies.add(cell.string)
        
#         print("found " + str(len(proxies)) + " proxies")

#         if not len(proxies) > 0:
#             attempt += 1
#             if attempt >= 10:
#                 raise requests.ConnectionError("Failed to \
#                     retrieve any proxy after several \
#                     attempts, check your connection status")
#             time.sleep(0.5)
#         else:
#             break

#     return proxies

def ParseReviews(asin,prox):
    # This script has only been tested with Amazon.com
    amazon_url  = 'http://www.amazon.com/dp/'+asin
    # Add some recent user agent to prevent amazon from blocking the request 
    # Find some chrome user agent strings  here https://udger.com/resources/ua-list/browser-detail?browser=Chrome
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'}
    try:
        for i in range(5):
            #print(type(ua.random))
            #print(type(random.sample(prox, 1 )))
            response = requests.get(amazon_url, timeout =30,
				        headers = {'User-Agent':ua.random})
            if response.status_code == 404:
                return {"url": amazon_url, "error": "page not found"}
            if response.status_code != 200:
                continue
            
            # Removing the null bytes from the response.
            cleaned_response = response.text.replace('\x00', '')
            
            parser = html.fromstring(cleaned_response)
            XPATH_AGGREGATE = '//span[@id="acrCustomerReviewText"]'
            XPATH_REVIEW_SECTION_1 = '//div[contains(@id,"reviews-summary")]'
            XPATH_REVIEW_SECTION_2 = '//div[@data-hook="review"]'
            XPATH_AGGREGATE_RATING = '//table[@id="histogramTable"]//tr'
            XPATH_PRODUCT_NAME = '//h1//span[@id="productTitle"]//text()'
            XPATH_PRODUCT_PRICE = '//span[@id="priceblock_ourprice"]/text()'
            
            X_PATH_PR_DES='.//*[@id="productDescription"]/p/text()'
            prod_desc=parser.xpath(X_PATH_PR_DES)
            print(prod_desc)

            raw_product_price = parser.xpath(XPATH_PRODUCT_PRICE)
            raw_product_name = parser.xpath(XPATH_PRODUCT_NAME)
            total_ratings  = parser.xpath(XPATH_AGGREGATE_RATING)
            reviews = parser.xpath(XPATH_REVIEW_SECTION_1)

            product_price = ''.join(raw_product_price).replace(',', '')
            product_name = ''.join(raw_product_name).strip()

            if not reviews:
                reviews = parser.xpath(XPATH_REVIEW_SECTION_2)
            ratings_dict = {}
            reviews_list = []

            # Grabing the rating  section in product page
            for ratings in total_ratings:
                extracted_rating = ratings.xpath('./td//a//text()')
                if extracted_rating:
                    rating_key = extracted_rating[0] 
                    raw_raing_value = extracted_rating[1]
                    rating_value = raw_raing_value
                    if rating_key:
                        ratings_dict.update({rating_key: rating_value})
            
            # Parsing individual reviews
            for review in reviews:
                XPATH_RATING  = './/i[@data-hook="review-star-rating"]//text()'
                XPATH_REVIEW_HEADER = './/a[@data-hook="review-title"]//text()'
                XPATH_REVIEW_POSTED_DATE = './/span[@data-hook="review-date"]//text()'
                XPATH_REVIEW_TEXT_1 = './/div[@data-hook="review-collapsed"]//text()'
                XPATH_REVIEW_TEXT_2 = './/div//span[@data-action="columnbalancing-showfullreview"]/@data-columnbalancing-showfullreview'
                XPATH_REVIEW_COMMENTS = './/span[@data-hook="review-comment"]//text()'
                XPATH_AUTHOR = './/span[contains(@class,"profile-name")]//text()'
                XPATH_REVIEW_TEXT_3 = './/div[contains(@id,"dpReviews")]/div/text()'
                
                
                
                raw_review_author = review.xpath(XPATH_AUTHOR)
                raw_review_rating = review.xpath(XPATH_RATING)
                raw_review_header = review.xpath(XPATH_REVIEW_HEADER)
                raw_review_posted_date = review.xpath(XPATH_REVIEW_POSTED_DATE)
                raw_review_text1 = review.xpath(XPATH_REVIEW_TEXT_1)
                raw_review_text2 = review.xpath(XPATH_REVIEW_TEXT_2)
                raw_review_text3 = review.xpath(XPATH_REVIEW_TEXT_3)

                # Cleaning data
                author = ' '.join(' '.join(raw_review_author).split())
                review_rating = ''.join(raw_review_rating).replace('out of 5 stars', '')
                review_header = ' '.join(' '.join(raw_review_header).split())

                try:
                    review_posted_date = dateparser.parse(''.join(raw_review_posted_date)).strftime('%d %b %Y')
                except:
                    review_posted_date = None
                review_text = ' '.join(' '.join(raw_review_text1).split())

                # Grabbing hidden comments if present
                if raw_review_text2:
                    json_loaded_review_data = json.loads(raw_review_text2[0])
                    json_loaded_review_data_text = json_loaded_review_data['rest']
                    cleaned_json_loaded_review_data_text = re.sub('<.*?>', '', json_loaded_review_data_text)
                    full_review_text = review_text+cleaned_json_loaded_review_data_text
                else:
                    full_review_text = review_text
                if not raw_review_text1:
                    full_review_text = ' '.join(' '.join(raw_review_text3).split())

                raw_review_comments = review.xpath(XPATH_REVIEW_COMMENTS)
                review_comments = ''.join(raw_review_comments)
                review_comments = re.sub('[A-Za-z]', '', review_comments).strip()
                
                review_dict = {
                                    #'review_comment_count': review_comments,
                                    'review_text': full_review_text,
                                    #'review_posted_date': review_posted_date,
                                    #'review_header': review_header,
                                    'review_rating': review_rating,
                                    #'review_author': author

                                }
            
                
                reviews_list.append(review_dict)

            data = {
                        'ratings': ratings_dict,
                        'proddesc':prod_desc,
                        'reviews': reviews_list,
                        #'url': amazon_url,
                        'name': product_name,
                        'price': product_price
                    
                    }
            return data, True
    except requests.ConnectionError as e:
        print("OOPS!! Connection Error. Make sure you are connected to Internet. Technical Details given below.\n")

    return {"error": "failed to process the page", "url": amazon_url}, False
            
def ReadAsin(asin,proxies):
    # AsinList = csv.DictReader(open(os.path.join(os.path.dirname(__file__),"Asinfeed.csv")))

    url = "http://www.amazon.com/dp/"+asin
    #print ("Processing: "+url)
    extracted_data,flag=ParseReviews(asin,proxies)
    sleep(15)
    
    return extracted_data,flag
 
 
if __name__ == "__main__":
    asins=[]
#    g = gzip.open("/home/snehal/Downloads/qa_Electronics.json.gz", 'r') 
#    json.load(g.decode("utf-8"))
#    with gzip.open("/home/snehal/Downloads/qa_Electronics.json.gz", "rb") as f:
#        d = json.loads(f.read().decode("ascii"))
#    extracted_data=[]
#    count=0
#    for l in g:
##        count+=1
##        print(l['asin'])
#        a=str(l)
#        b=json.load(l)
#        b['asin']
##        if 'product/productId' in a:
##            prod_id=(a.split(':')[1]).split('\\n')[0][1:]
##        elif 'product/description' in a:
##            prod_desc=(a.split(':',1)[1]).split('\\n')[0][1:]
##        if ((count+1)%3 == 0):
##            extracted_data.append(ReadAsin(prod_id,prod_desc))
#  
#    f=open('reviews_for_prod_desc_id.json','w')
#    json.dump(extracted_data,f,indent=4)
    


    def parse(path):
      g = gzip.open(path, 'rb')
      for l in g:
        yield eval(l)
    
    def getDF(path):
      i = 0
      df = {}
      for d in parse(path):
        df[i] = d
        i += 1
      return pd.DataFrame.from_dict(df, orient='index')

    category_list = sorted(os.listdir('Categorical data/'))
    extracted_data={}
    #proxies = get_proxies()
    proxies = ['127.0.0.1']
    print(proxies)
    for i in range(0,4):
        c = 0
        
        df = getDF("Categorical data/"+category_list[i])
        path =os.path.join("Review_pickles",category_list[i])
        if not os.path.exists(path):
            os.makedirs(path)
        asins=list(df.iloc[:,1])
        asin=list(set(asins))
        for val in asin:
            
            #product_dec_txt = open(category_list[i] +'product_desc_text.txt', 'a+')
            d,Flag = ReadAsin(val,proxies)
            extracted_data[val]= d
            with open(os.path.join(path,val +'.pickle'), 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #str1 = '%s : %s' %(val, str(d))
            #product_dec_txt.write(str1)
            #product_dec_txt.write('\n')
            c += 1
            if c % 20 == 0:
                sleep(60)
            
            #product_dec_txt.close()
                
        data= pd.DataFrame.from_dict(extracted_data, orient='index')
        pd.DataFrame.to_csv(data, category_list[i]+"_rev_des.csv")
       
    
        
    
#    f=open('reviews_for_prod_desc_id.json','w')
#    json.dump(extracted_data,f,indent=4)
    
 
        
