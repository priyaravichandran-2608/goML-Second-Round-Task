import pandas as pd
import re
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
file_path = "problem_2/AirlineReviews.csv"
df = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully!")

# Ensure necessary columns exist
if "Review" not in df.columns or "OverallScore" not in df.columns:
    raise ValueError("Dataset must contain 'Review' and 'OverallScore' columns.")

# Drop missing values
df = df[['Review', 'OverallScore']].dropna()

# Convert 'OverallScore' to 'Sentiment'
df["Sentiment"] = df["OverallScore"].apply(lambda x: "positive" if x >= 5 else "negative")

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

# Apply text cleaning
df["Cleaned_Review"] = df["Review"].apply(clean_text)

# Handle class imbalance
df_majority = df[df.Sentiment == "positive"]
df_minority = df[df.Sentiment == "negative"]

df_minority_upsampled = resample(df_minority, 
                                 replace=True, 
                                 n_samples=len(df_majority), 
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

df_balanced.to_csv("problem_2/cleaned_reviews.csv", index=False)
print("✅ Preprocessed dataset saved as 'cleaned_reviews.csv'!")

# Convert text data to numerical format
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df_balanced["Cleaned_Review"])
y = df_balanced["Sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression(class_weight="balanced", C=0.5)
model.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(model, "problem_2/sentiment_model.pkl")
joblib.dump(vectorizer, "problem_2/tfidf_vectorizer.pkl")

print("✅ Model trained and saved successfully!")