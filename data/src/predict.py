import joblib
from data_preprocessing import preprocess_text

def predict_spam(message):
    """Loads the trained model and predicts if the message is spam or not."""
    model = joblib.load("models/spam_classifier.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    
    # Preprocess input message
    cleaned_message = preprocess_text(message)
    
    # Convert to numerical features
    message_vector = vectorizer.transform([cleaned_message]).toarray()
    
    # Predict
    prediction = model.predict(message_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
if __name__ == "__main__":
    message = input("Enter an SMS message: ")
    print(f"Prediction: {predict_spam(message)}")
