from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(messages):
    """Converts text messages into numerical vectors using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(messages).toarray()
    return X, vectorizer
