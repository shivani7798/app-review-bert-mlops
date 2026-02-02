# app/preprocess.py

import re
import nltk
from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 0]
    return " ".join(tokens)
