import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load existing CSV datasets
df1 = pd.read_csv("combined_emotion.csv")
df2 = pd.read_csv("combined_sentiment_data.csv")

# Print dataset sizes before merging
print(f"➡ Combined Emotion Dataset: {len(df1)} rows")
print(f"➡ Combined Sentiment Dataset: {len(df2)} rows")

# Merge only df1 and df2
df = pd.concat([df1, df2], ignore_index=True)
print(f"➡ Total rows in final dataset before cleanup: {len(df)}")

# Show missing values before cleanup
print("Missing values before cleanup:\n", df.isnull().sum())

# Drop rows with missing sentiments
df.dropna(subset=["sentiment"], inplace=True)

# Ensure sentiment column is lowercase before mapping
df["sentiment"] = df["sentiment"].str.lower()

# Fill missing 'emotion' column (if applicable) and drop it if needed
if "emotion" in df.columns:
    df["sentiment"] = df["sentiment"].fillna(df["emotion"])
    df.drop(columns=["emotion"], inplace=True)

# Keep only valid sentiment values
valid_sentiments = ["negative", "neutral", "positive"]
df = df[df["sentiment"].isin(valid_sentiments)]

# Convert sentiment labels to numerical values
df["sentiment"] = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})

# Debugging: Check if all sentiment types are present
print("Unique Sentiments After Mapping:", df["sentiment"].unique())

# Function to clean text
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Keep .,!? but remove everything else
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces
    return text

# Apply text cleaning
df["sentence"] = df["sentence"].apply(clean_text)

# Show dataset size after cleanup
print("➡ Dataset size after cleanup:", len(df))
print(df.head(10))  # Debugging first few rows

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

print("✅ Model training complete. Saved as sa2_model.pkl")
print(f"✅ Total Samples After Merging: {len(df)}")

# Debugging: Test predictions on sample sentences
sample_texts = ["I love this product!", "This is bad.", "It's okay, nothing special."]
print("Sample Predictions:", model.predict(sample_texts))