"""Unit tests for the NewsPreprocessor class."""

import pytest
from src.preprocessor import NewsPreprocessor

@pytest.fixture
def preprocessor():
    return NewsPreprocessor()

def test_clean_text_removes_html(preprocessor):
    text = "<html><body>Hello world</body></html>"
    cleaned = preprocessor.clean_text(text)
    assert cleaned == "hello world"

def test_clean_text_removes_urls(preprocessor):
    text = "Check out this link http://example.com/foo and this one www.test.org"
    cleaned = preprocessor.clean_text(text)
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert "check out this link  and this one" in cleaned

def test_tokenize_returns_list(preprocessor):
    text = "this is a test sentence"
    tokens = preprocessor.tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) == 5

def test_remove_stopwords_reduces_count(preprocessor):
    tokens = ["this", "is", "a", "test", "sentence"]
    filtered = preprocessor.remove_stopwords(tokens)
    assert len(filtered) < len(tokens)
    assert "is" not in filtered

def test_preprocess_returns_string(preprocessor):
    text = "<html>Check out http://example.com! It's very interesting.</html>"
    result = preprocessor.preprocess(text)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "http" not in result
