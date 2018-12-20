# README file for AQUAA - Amazon Question Answering Assistant

# Contents:
	# Configuration
		- config.py : Contains values and hyperparameters required for the network

	# Data Preparation
		- scraper_reviews_new.py : Scrapes amazon product page for name, descriptions, reviews and price 
		- DataProc.py : Creates a structured dataset from the scraped Amazon dataset and the 
						question answer dataset to combine all information into a pickle file
						across different categories and different product ids (Categorical_data is the existing Amazon Q-A data)
						(Review_data is the scraped amazon data containing the reviews, product description, price)
						and extracts mined opinions from the reviews
		- make_new_qa.py : Used for generating custom question-answer pairs related to sentiments 
							 extracted from opinion mining from the Amazon reviews
		- split_data.py : splits the data into training and testing set
		- new_data_loader.py : Contains the helper functions to process the data 
								for it to be usable by the network, like embeddings, creating vocabulary dictionary
								marking out the ground truth answers


	# Model training
		- model.py : Contains the MatchLSTM-PointerNet architecture for selecting start and end point for 
					 answers from a passage
		- train.py : Training process of the network for given parameters 
		- inference.py : Used for infering the output(answers) based on the checkpoint weights

	# Sentiment Analysis
		- sentiment_analysis.py : Opinion mining of the sentiments of Amazon reviews
								  produces a histogram, shows division of positive 
								  and negative sentiments


# Dependencies:
	- Python 2.7+/3.6+
	- run pip install -r requirements_train.txt (For training, python 2 dependencies)
	- run pip install -r requirements_test.txt (For data preprocessing and inference, python 3 dependencies)


# Steps to run the program:
	-------------------- Training -----------------------
	# run training in a Python 2.7+ environment
	- python train.py

	------------------- Inference -----------------------
	# run inference in a Python 3.6+ environment
	- python inference.py

	--------------- Sentiment Analysis -------------
	# run sentiment analysis in a Python 3.6+ environment
	- python sentiment_analysis.py

	---------------- Data Preparation -------------------
	# Run these steps only if you want to generate the processed data 
	# else it is already present in the data folder
	# run data preparation in a Python 3.6+ environment
	- python DataProc.py Dataset/Categorical_data Dataset/Review_data 
	- python make_new_qa.py
	- python split_data.py
	- python new_data_loader.py