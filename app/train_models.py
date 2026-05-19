"""Standalone script to train, evaluate, and save both ML models."""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.topic_classifier import TopicClassifier
from src.models.fake_detector import FakeNewsDetector

MODELS_DIR = "saved_models"
AG_NEWS_PATH = "data/samples/ag_news_sample.csv"
WELFAKE_PATH = "data/samples/welfake_sample.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: str, name: str) -> pd.DataFrame | None:
    """
    Load a CSV dataset from disk, with a graceful error if missing.

    Args:
        path (str): File path to the CSV file.
        name (str): Human-readable dataset name for error messages.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if loading failed.
    """
    if not os.path.exists(path):
        print(f"[ERROR] {name} not found at '{path}'.")
        print(f"        Run: python app/data_download.py first to generate sample datasets.")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[OK] Loaded {name}: {len(df)} rows from '{path}'")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read {name}: {e}")
        return None


def train_topic_classifier(df: pd.DataFrame) -> None:
    """
    Train and evaluate the TopicClassifier, then save to disk.

    Args:
        df (pd.DataFrame): DataFrame with 'text', 'label', and 'label_name' columns.
    """
    print("\n" + "="*60)
    print("  Training TopicClassifier")
    print("="*60)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df['label'], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    clf = TopicClassifier()
    clf.train(train_df)

    # Evaluate
    accuracy = clf.get_accuracy(test_df)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Detailed report
    texts = test_df['text'].astype(str).tolist()
    true_labels = test_df['label'].tolist()
    X_test = clf.extractor.transform(texts)
    pred_labels = clf.model.predict(X_test)
    target_names = [clf.label_names[l] for l in sorted(clf.label_names.keys())]
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    clf.save(MODELS_DIR)


def train_fake_detector(df: pd.DataFrame) -> None:
    """
    Train and evaluate the FakeNewsDetector, then save to disk.

    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'label' columns (0=Fake, 1=Real).
    """
    print("\n" + "="*60)
    print("  Training FakeNewsDetector")
    print("="*60)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df['label'], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    detector = FakeNewsDetector()
    detector.train(train_df)

    # Evaluate
    accuracy = detector.get_accuracy(test_df)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Detailed report
    texts = test_df['text'].astype(str).tolist()
    true_labels = test_df['label'].tolist()
    X_test = detector.extractor.transform(texts)
    pred_labels = detector.model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=['Fake', 'Real']))

    detector.save(MODELS_DIR)


def main():
    """
    Orchestrate the full training pipeline for all models.
    Loads sample datasets, trains each model, prints evaluation, and saves artifacts.
    """
    print("\n" + "#"*60)
    print("  News Classifier — Model Training Pipeline")
    print("#"*60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Topic Classifier (AG News) ---
    ag_df = load_dataset(AG_NEWS_PATH, "AG News")
    if ag_df is not None:
        train_topic_classifier(ag_df)
    else:
        print("[SKIP] Skipping TopicClassifier training.")

    # --- Fake News Detector (WELFake) ---
    welfake_df = load_dataset(WELFAKE_PATH, "WELFake")
    if welfake_df is not None:
        train_fake_detector(welfake_df)
    else:
        print("[SKIP] Skipping FakeNewsDetector training.")

    print("\n" + "#"*60)
    print("  Training complete. Models saved to:", MODELS_DIR)
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
