"""Text preprocessing and cleaning utilities."""

import re
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

class NewsPreprocessor:
    """A class to handle text preprocessing for the News Classifier project."""

    def __init__(self):
        """Initialize the preprocessor with Spacy and NLTK resources."""
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """
        Strips HTML tags, removes URLs, removes special characters, and lowercases the text.
        
        Args:
            text (str): The raw input string.
            
        Returns:
            str: The cleaned and lowercased string.
        """
        # Strip HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Lowercase
        return text.lower().strip()

    def tokenize(self, text: str) -> list:
        """
        Tokenizes text using NLTK's word_tokenize.
        
        Args:
            text (str): The cleaned text string.
            
        Returns:
            list: A list of tokens.
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        """
        Removes standard English stopwords from a list of tokens.
        
        Args:
            tokens (list): A list of word tokens.
            
        Returns:
            list: Tokens with stopwords removed.
        """
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens: list) -> list:
        """
        Lemmatizes a list of tokens using Spacy.
        
        Args:
            tokens (list): A list of word tokens.
            
        Returns:
            list: Lemmatized tokens.
        """
        # Join tokens back to a string for Spacy processing
        doc = self.nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]

    def preprocess(self, text: str) -> str:
        """
        Chains all preprocessing steps: cleaning, tokenizing, stopword removal, and lemmatization.
        
        Args:
            text (str): The raw input string.
            
        Returns:
            str: The fully preprocessed and lemmatized string.
        """
        if not text or not isinstance(text, str):
            return ""
            
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize(filtered_tokens)
        
        return " ".join(lemmatized_tokens)

if __name__ == "__main__":
    sample = "<html><body>Breaking News: The new AI model is released at https://example.com! It's simply amazing and 100% better.</body></html>"
    print(f"Original: {sample}")
    
    preprocessor = NewsPreprocessor()
    processed = preprocessor.preprocess(sample)
    
    print(f"Processed: {processed}")
