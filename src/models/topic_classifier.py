"""Topic classification model using Logistic Regression and TF-IDF features."""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.features import FeatureExtractor


class TopicClassifier:
    """
    A news topic classifier using TF-IDF features and Logistic Regression.

    Classifies news articles into categories such as Politics, Sports,
    Business, and Technology. Supports saving and loading model artifacts
    to/from disk for reuse at inference time.
    """

    def __init__(self):
        """Initialize the TopicClassifier with a FeatureExtractor and untrained model."""
        self.extractor = FeatureExtractor()
        self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
        self.label_names: dict = {}   # int -> str mapping
        self.int_labels: dict = {}    # str -> int mapping
        self._is_fitted = False

    def train(self, df: pd.DataFrame) -> None:
        """
        Fit the FeatureExtractor and Logistic Regression model on the training data.

        Args:
            df (pd.DataFrame): A DataFrame with 'text' (str) and 'label' (int) columns.
                               Optionally, a 'label_name' column is used to build the
                               label mapping; otherwise, labels are stringified.
        """
        texts = df['text'].astype(str).tolist()
        labels = df['label'].tolist()

        # Build label name mapping
        if 'label_name' in df.columns:
            mapping = df[['label', 'label_name']].drop_duplicates()
            self.label_names = dict(zip(mapping['label'], mapping['label_name']))
        else:
            unique_labels = sorted(df['label'].unique())
            self.label_names = {l: str(l) for l in unique_labels}
        self.int_labels = {v: k for k, v in self.label_names.items()}

        X = self.extractor.fit_transform(texts)
        self.model.fit(X, labels)
        self._is_fitted = True
        print(f"TopicClassifier trained on {len(texts)} samples, {len(self.label_names)} classes.")

    def predict(self, text: str) -> str:
        """
        Predict the topic label name for a single text string.

        Args:
            text (str): Raw news article text.

        Returns:
            str: The predicted topic label name (e.g., 'Sports', 'Technology').
        """
        X = self.extractor.transform([str(text)])
        int_label = self.model.predict(X)[0]
        return self.label_names.get(int_label, str(int_label))

    def predict_proba(self, text: str) -> dict:
        """
        Return the predicted class probabilities for a single text string.

        Args:
            text (str): Raw news article text.

        Returns:
            dict: A mapping of {label_name (str): probability (float)} for all classes.
        """
        X = self.extractor.transform([str(text)])
        proba_array = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        return {self.label_names.get(c, str(c)): float(p) for c, p in zip(classes, proba_array)}

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
        self.extractor.save(os.path.join(model_dir, 'topic_tfidf.joblib'))
        joblib.dump(self.model, os.path.join(model_dir, 'topic_lr.joblib'))
        joblib.dump(self.label_names, os.path.join(model_dir, 'topic_labels.joblib'))
        print(f"TopicClassifier saved to: {model_dir}/")

    def load(self, model_dir: str) -> None:
        """
        Load the vectorizer and classifier from the given directory.

        Args:
            model_dir (str): Directory path containing the saved model files.

        Raises:
            FileNotFoundError: If any expected model file is missing in the directory.
        """
        tfidf_path = os.path.join(model_dir, 'topic_tfidf.joblib')
        lr_path = os.path.join(model_dir, 'topic_lr.joblib')
        labels_path = os.path.join(model_dir, 'topic_labels.joblib')

        for p in [tfidf_path, lr_path, labels_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"TopicClassifier model file not found: {p}")

        self.extractor.load(tfidf_path)
        self.model = joblib.load(lr_path)
        self.label_names = joblib.load(labels_path)
        self.int_labels = {v: k for k, v in self.label_names.items()}
        self._is_fitted = True
        print(f"TopicClassifier loaded from: {model_dir}/")
