import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import preprocess_text
from feature_extraction import extract_features

# Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess messages
df['cleaned_message'] = df['message'].apply(preprocess_text)

# Extract features
X, vectorizer = extract_features(df['cleaned_message'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
