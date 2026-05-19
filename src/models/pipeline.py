"""End-to-end inference pipeline combining topic classification and fake news detection."""

import time

from src.models.topic_classifier import TopicClassifier
from src.models.fake_detector import FakeNewsDetector


class NewsPipeline:
    """
    An end-to-end inference pipeline that runs both the TopicClassifier
    and FakeNewsDetector on a given news article.

    Loads both models from disk on initialization. Returns a structured
    dictionary of results for each analyzed article, suitable for direct
    use in a Streamlit UI or REST API.
    """

    def __init__(self, models_dir: str = "saved_models"):
        """
        Load both trained models from the given directory.

        Args:
            models_dir (str): Path to the directory containing saved model files.
                              Defaults to 'saved_models'.
        """
        self.models_dir = models_dir
        self.topic_classifier = TopicClassifier()
        self.fake_detector = FakeNewsDetector()
        self._loaded = False

        try:
            self.topic_classifier.load(models_dir)
            self.fake_detector.load(models_dir)
            self._loaded = True
        except FileNotFoundError as e:
            print(f"[NewsPipeline] Warning: Could not load models. {e}")
            print("[NewsPipeline] Run app/train_models.py first to train and save models.")

    def analyze(self, text: str) -> dict:
        """
        Run the full inference pipeline on a single article.

        Performs topic classification and fake news detection simultaneously,
        returning a structured result dictionary.

        Args:
            text (str): Raw news article text to analyze.

        Returns:
            dict: A dictionary with keys:
                - 'topic' (str): Predicted topic category.
                - 'topic_conf' (float): Confidence score for the predicted topic.
                - 'verdict' (str): 'Real' or 'Fake'.
                - 'fake_conf' (float): Confidence that the article is Fake.
                - 'processing_ms' (int): Time taken for inference in milliseconds.
                - 'error' (str, optional): Present only if an error occurred.
        """
        if not self._loaded:
            return {
                "error": "Models are not loaded. Please run app/train_models.py first.",
                "topic": None,
                "topic_conf": None,
                "verdict": None,
                "fake_conf": None,
                "processing_ms": 0,
            }

        try:
            start_time = time.time()

            # Topic classification
            topic = self.topic_classifier.predict(text)
            topic_proba = self.topic_classifier.predict_proba(text)
            topic_conf = topic_proba.get(topic, 0.0)

            # Fake news detection
            verdict = self.fake_detector.predict(text)
            fake_proba = self.fake_detector.predict_proba(text)
            fake_conf = fake_proba.get('Fake', 0.0)

            processing_ms = int((time.time() - start_time) * 1000)

            return {
                "topic": topic,
                "topic_conf": round(topic_conf, 4),
                "verdict": verdict,
                "fake_conf": round(fake_conf, 4),
                "processing_ms": processing_ms,
            }

        except Exception as e:
            return {
                "error": f"Inference failed: {str(e)}",
                "topic": None,
                "topic_conf": None,
                "verdict": None,
                "fake_conf": None,
                "processing_ms": 0,
            }
