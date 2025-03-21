# Spam SMS Detection

## 📌 Project Overview
The objective of this project is to develop an **SMS classification model** that accurately distinguishes between spam and legitimate messages. The model is trained using a labeled dataset and applies **Natural Language Processing (NLP) techniques** to classify messages as **Spam or Not Spam**.

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, NLTK, Scikit-learn, Joblib
- **Machine Learning Model**: TF-IDF Vectorizer + Naïve Bayes Classifier

## 📂 Project Structure
```
📁 spam-sms-detection/
│-- 📂 data/              # Dataset files
│-- 📂 models/            # Trained models
│-- 📂 src/               # Source code
│   ├── data_preprocessing.py  # Text preprocessing functions
│   ├── model_training.py      # Model training script
│   ├── predict.py             # Prediction script
│-- requirements.txt      # Required dependencies
│-- README.md            # Project documentation
```

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/spam-sms-detection.git
cd spam-sms-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
Run the following command to train the model:
```bash
python src/model_training.py
```
After training, the model is saved in the `models/` directory.

### 4️⃣ Test the Prediction
Run the script and input an SMS message to test if it's spam:
```bash
python src/predict.py
```
Example:
```
Enter an SMS message: Congratulations! You won a free iPhone. Click here to claim now!
Prediction: Spam
```

## 🔍 Evaluation Criteria
- **Accuracy of the model** in detecting spam messages
- **Clean and structured code** with proper documentation
- **A well-documented README** with setup instructions

## 📝 Notes
- The dataset is preprocessed by **removing stopwords, stemming words, and applying TF-IDF vectorization**.
- The trained model is a **Naïve Bayes classifier**, which is effective for text classification tasks.
- Ensure the dataset is balanced to avoid bias in spam detection.

---
📌 **Author**: PULI LASYA REDDY  
📅 **Date**: March 2025  
