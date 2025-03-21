from src.model_training import *
from src.predict import predict_spam

if __name__ == "__main__":
    # Train model
    print("Training model...")
    train_model()

    # Test prediction
    sample_message = "Congratulations! You won a free gift card. Click here to claim now."
    print(f"Message: {sample_message}")
    print(f"Prediction: {predict_spam(sample_message)}")
