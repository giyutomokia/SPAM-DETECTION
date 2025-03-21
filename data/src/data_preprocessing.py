import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def preprocess_text(text):
    """Cleans text by removing non-alphabet characters, converting to lowercase, 
       removing stopwords, and applying stemming."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)
