# Text preprocessing functions

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download only once
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', ' ', text)

    # Remove hashtags (#topic)
    text = re.sub(r'#\w+', ' ', text)

    # Keep only alphabets (removes digits, punctuation, special chars, newline, etc.)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization: Break the text into individual words (called tokens)
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatization: Convert each word to its root/base form
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # Join tokens back converting into a single string
    return " ".join(tokens)