import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load model and vectorizer
model = joblib.load("problem_2/sentiment_model.pkl")
vectorizer = joblib.load("problem_2/tfidf_vectorizer.pkl")

# Load dataset
df = pd.read_csv("problem_2/cleaned_reviews.csv")

# Ensure there are no NaN values
df["Cleaned_Review"] = df["Cleaned_Review"].fillna("")

# Convert text to numerical format
X = vectorizer.transform(df["Cleaned_Review"])
y = df["Sentiment"]

# Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Make predictions
y_pred = model.predict(X_test)

# Print evaluation metrics
print("\n🔹 Model Evaluation Metrics 🔹")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
