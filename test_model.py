import joblib

# Load the trained model
model = joblib.load("sa2_model.pkl")

# Test sample sentences
test_sentences = [
    "I love this place, the food is amazing!",
    "The service was terrible, I will never come back.",
    "It was an okay experience, nothing special."
]

# Make predictions
predictions = model.predict(test_sentences)

# Convert back to labels
label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

# Print results
for sentence, pred in zip(test_sentences, predictions):
    print(f"Sentence: {sentence} → Predicted Sentiment: {label_mapping[pred]}")
