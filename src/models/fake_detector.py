"""Fake news detection model using Passive-Aggressive Classifier and TF-IDF features."""

import os
import joblib
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

from src.features import FeatureExtractor


class FakeNewsDetector:
    """
    A fake news detector using TF-IDF features and a Passive-Aggressive Classifier.

    Classifies news articles as either 'Real' or 'Fake'. The Passive-Aggressive
    Classifier is well suited for online and large-scale text classification tasks.
    Supports saving and loading model artifacts to/from disk.
    """

    LABELS = {0: 'Fake', 1: 'Real'}

    def __init__(self):
        """Initialize the FakeNewsDetector with a FeatureExtractor and untrained model."""
        self.extractor = FeatureExtractor()
        self.model = PassiveAggressiveClassifier(C=1.0, random_state=42, max_iter=1000)
        self.label_names = self.LABELS.copy()   # {0: 'Fake', 1: 'Real'}
        self._is_fitted = False

    def train(self, df: pd.DataFrame) -> None:
        """
        Fit the FeatureExtractor and Passive-Aggressive Classifier on the training data.

        Args:
            df (pd.DataFrame): A DataFrame with 'text' (str) and 'label' (int) columns.
                               'label' must be 0 for Fake and 1 for Real.
        """
        texts = df['text'].astype(str).tolist()
        labels = df['label'].tolist()

        X = self.extractor.fit_transform(texts)
        self.model.fit(X, labels)
        self._is_fitted = True
        print(f"FakeNewsDetector trained on {len(texts)} samples.")

    def predict(self, text: str) -> str:
        """
        Predict whether a single article is 'Real' or 'Fake'.

        Args:
            text (str): Raw news article text.

        Returns:
            str: Either 'Real' or 'Fake'.
        """
        X = self.extractor.transform([str(text)])
        int_label = int(self.model.predict(X)[0])
        return self.label_names.get(int_label, 'Unknown')

    def predict_proba(self, text: str) -> dict:
        """
        Return confidence scores for 'Real' and 'Fake' classes.

        The Passive-Aggressive Classifier does not natively produce probabilities,
        so the decision function score is converted to a [0,1] range using a
        sigmoid transformation.

        Args:
            text (str): Raw news article text.

        Returns:
            dict: {'Real': float, 'Fake': float} where values sum to ~1.0.
        """
        import math
        X = self.extractor.transform([str(text)])
        # decision_function gives a signed distance from the hyperplane
        score = self.model.decision_function(X)[0]
        # Sigmoid to convert to probability of being Real (label=1)
        prob_real = 1.0 / (1.0 + math.exp(-score))
        prob_fake = 1.0 - prob_real
        return {'Real': round(prob_real, 4), 'Fake': round(prob_fake, 4)}

    def get_accuracy(self, df: pd.DataFrame) -> float:
        """
        Compute accuracy on a labelled DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame with 'text' and 'label' columns.

        Returns:
            float: Accuracy score between 0.0 and 1.0.
        """
        texts = df['text'].astype(str).tolist()
        labels = df['label'].tolist()
        X = self.extractor.transform(texts)
        preds = self.model.predict(X)
        return accuracy_score(labels, preds)

    def save(self, model_dir: str) -> None:
        """
        Save the vectorizer and classifier to the given directory as joblib files.

        Args:
            model_dir (str): Directory path where model files will be written.
        """
        os.makedirs(model_dir, exist_ok=True)
        self.extractor.save(os.path.join(model_dir, 'fake_tfidf.joblib'))
        joblib.dump(self.model, os.path.join(model_dir, 'fake_pac.joblib'))
        print(f"FakeNewsDetector saved to: {model_dir}/")

    def load(self, model_dir: str) -> None:
        """
        Load the vectorizer and classifier from the given directory.

        Args:
            model_dir (str): Directory path containing the saved model files.

        Raises:
            FileNotFoundError: If any expected model file is missing in the directory.
        """
        tfidf_path = os.path.join(model_dir, 'fake_tfidf.joblib')
        pac_path = os.path.join(model_dir, 'fake_pac.joblib')

        for p in [tfidf_path, pac_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"FakeNewsDetector model file not found: {p}")

        self.extractor.load(tfidf_path)
        self.model = joblib.load(pac_path)
        self._is_fitted = True
        print(f"FakeNewsDetector loaded from: {model_dir}/")
