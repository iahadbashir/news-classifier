"""Feature extraction utilities using TF-IDF vectorization."""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """
    A wrapper around scikit-learn's TfidfVectorizer for consistent feature extraction.

    Uses bigrams and sublinear TF scaling for improved text representation.
    Supports serialization so the fitted vectorizer can be reused at inference time.
    """

    def __init__(self, max_features: int = 50000, ngram_range: tuple = (1, 2), sublinear_tf: bool = True):
        """
        Initialize the FeatureExtractor with TF-IDF parameters.

        Args:
            max_features (int): Maximum number of features (vocabulary size).
            ngram_range (tuple): Range of n-gram sizes to extract.
            sublinear_tf (bool): Apply sublinear (log) scaling to term frequencies.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',
        )

    def fit_transform(self, texts: list):
        """
        Fit the vectorizer on texts and return the transformed sparse matrix.

        Args:
            texts (list): A list of raw text strings for training.

        Returns:
            scipy.sparse matrix: TF-IDF feature matrix of shape (n_samples, n_features).
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list):
        """
        Transform texts using an already-fitted vectorizer.

        Args:
            texts (list): A list of raw text strings to transform.

        Returns:
            scipy.sparse matrix: TF-IDF feature matrix of shape (n_samples, n_features).

        Raises:
            Exception: If the vectorizer has not been fitted yet.
        """
        return self.vectorizer.transform(texts)

    def save(self, path: str) -> None:
        """
        Serialize and save the fitted vectorizer to disk using joblib.

        Args:
            path (str): File path where the vectorizer will be saved (e.g., 'saved_models/tfidf.joblib').
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"FeatureExtractor saved to: {path}")

    def load(self, path: str) -> None:
        """
        Load a previously saved vectorizer from disk.

        Args:
            path (str): File path to the saved vectorizer.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vectorizer file not found at: {path}")
        self.vectorizer = joblib.load(path)
        print(f"FeatureExtractor loaded from: {path}")
