import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained model and vectorizer
model = joblib.load("problem_2/sentiment_model.pkl")
vectorizer = joblib.load("problem_2/tfidf_vectorizer.pkl")

# Text Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Function to predict sentiment
def predict_sentiment(review):
    cleaned_review = clean_text(review)
    transformed_review = vectorizer.transform([cleaned_review])
    prediction = model.predict(transformed_review)[0]
    return prediction

# Get real-time user input
user_review = input("Enter your review: ")
sentiment = predict_sentiment(user_review)
print(f"Predicted Sentiment: {sentiment}")
