# Sentiment Analysis on Airline Reviews

## Overview  
This project builds a sentiment analysis model that classifies airline reviews as positive or negative based on textual input. The model is trained using Logistic Regression with TF-IDF vectorization for text representation.


## Installation  
To set up the environment and install dependencies, follow these steps:  


## Install Dependencies 
Ensure you have Python installed (version 3.8 or later). Then, install the required libraries:  
pip install -r requirements.txt

## Download NLTK Resources  
Run the following script to download required NLP resources:  
python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")


## Running the Code  

### Train the Sentiment Model
Run the following command to train and save the model:  

python main1.py

This script will:  
✅ Load and preprocess the dataset  
✅ Train a Logistic Regression model  
✅ Save the trained model and TF-IDF vectorizer  

### Predict Sentiment for Real-Time Input
Run the following command to test the model with a custom review:  
python predict_sentiment.py

Enter a review, and the model will return "Positive" or "Negative" sentiment.


## File Structure  

/sentiment-analysis
│── problem_2/
│   ├── AirlineReviews.csv  # Dataset
│   ├── cleaned_reviews.csv  # Processed dataset
│   ├── sentiment_model.pkl  # Trained model
│   ├── tfidf_vectorizer.pkl  # Vectorizer
│   ├── main.py  # Model training script
│   └── predict.py  # Sentiment prediction script
│── README.md  # Documentation
│── requirements.txt  # Dependencies


## Future Enhancements  
Support for neutral sentiment  
Deploy as a Flask API  
Integrate with LangChain for multi-agent reasoning




