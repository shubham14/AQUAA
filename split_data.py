import pickle
import numpy as np
import copy
import os
from config import Config as cfg
# Splits 80% train, 20% test per category
# train_size : proportion of data to be split into train; must be between 0 and 1!
def split_data(og_data, train_size):
    # Deep copy because we don't want to modify original values
    data = copy.deepcopy(og_data)

    asins = []
    asin_to_cat = {}

    # Create a dict to map asin back to category
    # So that we can extract the values later
    for category in data:
        for asin in data[category]:
            asins.append(asin)
            asin_to_cat[asin] = category

    # Find how much data we have
    data_length = len(asins)
    # Calculate how much data will be in the train set
    train_length = int(train_size * data_length)

    np.random.shuffle(asins)
    # Select asins randomly without replacement from the data until we have train_length data
    train_asins = np.random.choice(asins, size=train_length, replace=False)

    train_asin = {}
    test_asin = {}

    # Construct a train dict with only the values of the asins in train_asins
    # Then remove that value from our list of all asins
    # Our dataset will then only contain test asins
    for asin in train_asins:
        train_asin[asin] = data[asin_to_cat[asin]][asin]
        asins.remove(asin)
    
    # Construct a test dict with the values of the remaining asins in asins
    for asin in asins:
        test_asin[asin] = data[asin_to_cat[asin]][asin]

    return train_asin, test_asin

# Splits 80% train, 20% test per category
# train_size : proportion of data to be split into train; must be between 0 and 1!
def split_data_cat(og_data, train_size):
    # Deep copy because we don't want to modify original values
    data = copy.deepcopy(og_data)
    train = {}
    test = {}

    # For each category
    for category in data:
        # Find how much data we have
        data_length = len(data[category])
        # Calculate how much data will be in the train set
        train_length = int(train_size * data_length)

        asins = list(data[category].keys())
        np.random.shuffle(asins)
        # Select asins randomly without replacement from the data until we have train_length data
        train_asins = np.random.choice(asins, size=train_length, replace=False)

        test_asin = data[category]
        train_asin = {}

        # Construct a train dict with only the values of the asins in train_asins
        # Then remove that value from our copied dataset
        # Our dataset will then only contain test values
        for asin in train_asins:
            train_asin[asin] = data[category][asin]
            del test_asin[asin]

        train[category] = train_asin
        test[category] = test_asin

    return train, test

#Lowercase all text
def pre_process(data):
    categories = {}
    for category in data:
        asins = {}
        for asin in data[category]:
            context = data[category][asin]["context"]
            context = context.lower()
            qa = []
            for sentences in data[category][asin]["QA"]:
                sen = []
                
                for sentence in sentences[:-1]:
                    sentence = sentence.lower() 
                    sen.append(sentence)
                qa.append(sen)
            
            asins[asin] =  {
                    "context" : context,
                    "QA" : qa
                }

        categories[category] = asins
    return categories
    

def main():
    # Open all data files
    filepath = os.path.join(cfg.DATA_DIR, "amz_new_qa_pairs.pkl")
    
    # Cat means that the data is seperated by category
    train_cat_filepath = os.path.join(cfg.DATA_DIR, "train_cat_minimal.pkl")
    test_cat_filepath = os.path.join(cfg.DATA_DIR, "test_cat.pkl")
    
    #train_filepath = 'Dataset/train.pkl'
    #test_filepath = 'Dataset/test.pkl'
    
    data_file = open(filepath, 'rb')
    
    train_cat_file = open(train_cat_filepath, 'wb')
    test_cat_file = open(test_cat_filepath, 'wb')

    #train_file = open(train_filepath, 'wb')
    #test_file = open(test_filepath, 'wb')

    data = pickle.load(data_file)
    data = pre_process(data)
    
    # Splits the data by category
    train_cat, test_cat = split_data_cat(data, train_size=0.1)

    # Splits the data overall
    #train, test = split_data(data, train_size=0.1)
   
    # Write data out to files
    pickle.dump(train_cat, train_cat_file,protocol=2)
    pickle.dump(test_cat, test_cat_file,protocol=2)

    #pickle.dump(train, train_file)
    #pickle.dump(test, test_file)

    # Close all files
    data_file.close()
    train_cat_file.close()
    test_cat_file.close()
    #train_file.close()
    #test_file.close()

    print("Done Splitting Data!")

if __name__ == "__main__":
    main()