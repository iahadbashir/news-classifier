"""Unit tests for the News models."""

import pytest
from unittest.mock import patch
from src.models.pipeline import NewsPipeline

@patch('src.models.pipeline.TopicClassifier')
@patch('src.models.pipeline.FakeNewsDetector')
def test_pipeline_result_has_required_keys(mock_fake, mock_topic):
    # Mock the load methods so it doesn't try to load files
    pipeline = NewsPipeline(models_dir="dummy_dir")
    pipeline._loaded = True # Force loaded state
    
    # Mock predictions
    pipeline.topic_classifier.predict.return_value = "Technology"
    pipeline.topic_classifier.predict_proba.return_value = {"Technology": 0.9}
    pipeline.fake_detector.predict.return_value = "Real"
    pipeline.fake_detector.predict_proba.return_value = {"Real": 0.85, "Fake": 0.15}
    
    result = pipeline.analyze("Test article text")
    
    expected_keys = {"topic", "topic_conf", "verdict", "fake_conf", "processing_ms"}
    assert set(result.keys()) == expected_keys

@patch('src.models.pipeline.TopicClassifier')
@patch('src.models.pipeline.FakeNewsDetector')
def test_verdict_is_real_or_fake(mock_fake, mock_topic):
    pipeline = NewsPipeline(models_dir="dummy_dir")
    pipeline._loaded = True
    
    pipeline.topic_classifier.predict.return_value = "Politics"
    pipeline.topic_classifier.predict_proba.return_value = {"Politics": 0.7}
    pipeline.fake_detector.predict.return_value = "Fake"
    pipeline.fake_detector.predict_proba.return_value = {"Real": 0.2, "Fake": 0.8}
    
    result = pipeline.analyze("Test article")
    assert result["verdict"] in ["Real", "Fake"]

@patch('src.models.pipeline.TopicClassifier')
@patch('src.models.pipeline.FakeNewsDetector')
def test_topic_conf_is_float_between_0_and_1(mock_fake, mock_topic):
    pipeline = NewsPipeline(models_dir="dummy_dir")
    pipeline._loaded = True
    
    pipeline.topic_classifier.predict.return_value = "Sports"
    pipeline.topic_classifier.predict_proba.return_value = {"Sports": 0.95}
    pipeline.fake_detector.predict.return_value = "Real"
    pipeline.fake_detector.predict_proba.return_value = {"Real": 0.99, "Fake": 0.01}
    
    result = pipeline.analyze("Test article")
    assert isinstance(result["topic_conf"], float)
    assert 0.0 <= result["topic_conf"] <= 1.0
