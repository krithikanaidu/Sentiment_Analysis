import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load datasets
df1 = pd.read_csv("combined_emotion.csv")
df2 = pd.read_csv("combined_sentiment_data.csv")

# Merge datasets
df = pd.concat([df1, df2], ignore_index=True)

# Check missing values
print("Missing values before cleanup:\n", df.isnull().sum())

# Drop rows where 'sentiment' is missing
df = df.dropna(subset=["sentiment"])

# Fill missing 'emotion' with "neutral" (Optional)
df = df.assign(emotion=df["emotion"].fillna("neutral"))


# Ensure sentiment labels are valid
valid_sentiments = ["negative", "neutral", "positive"]
df = df[df["sentiment"].isin(valid_sentiments)]

# Map sentiments to numerical labels
sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df["sentiment"] = df["sentiment"].map(sentiment_mapping)

# Preprocessing function
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text.strip()

# Apply text cleaning
df["sentence"] = df["sentence"].apply(clean_text)

# Check dataset size after cleaning
print("Dataset size after cleanup:", len(df))

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["sentence"], df["sentiment"], test_size=0.2, random_state=42)

# Create a TF-IDF + Naive Bayes Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()), 
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "sa2_model.pkl")

print("Model training complete. Saved as sa2_model.pkl")
