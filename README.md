# News Classifier System

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An automated ML pipeline for categorizing news topics and identifying misinformation. 

This project processes raw news articles or screenshots using NLP and OCR to simultaneously predict the article's topic and verify its credibility using ensemble classical machine learning models.

## Features
* **Topic Classification:** Automatically categorizes text into 6 distinct categories (Politics, Sports, Business, Technology, Entertainment, Health).
* **Fake News Detection:** Analyzes linguistic patterns to flag potential misinformation.
* **OCR Image Input:** Extracts and processes text directly from uploaded screenshots of news articles.
* **Classical ML & Transformer Modes:** Uses high-efficiency TF-IDF and Logistic Regression pipelines with an optional Transformer-based mode for deep semantic analysis.
* **Interactive Streamlit UI:** A clean, user-friendly web interface for testing the models in real-time.

## System Architecture

```text
               [Input]
             Text or Image
                   │
                   ▼
           ┌───────────────┐
           │  Preprocessor │ (OCR, HTML Stripping, Tokenization, Lemmatization)
           └───────┬───────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Feature Extractor  │ (TF-IDF Vectorization)
        └──────────┬──────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌──────────────────┐
│ TopicClassifier │ │ FakeNewsDetector │
└────────┬────────┘ └────────┬─────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
             ┌──────────┐
             │ Pipeline │ (Aggregation)
             └─────┬────┘
                   │
                   ▼
               [Results]
        (Topic & Authenticity)
```

## Model Performance

| Task | Model | Dataset | Accuracy |
|------|-------|---------|----------|
| Topic Classification | TF-IDF + Logistic Regression | AG News | 88-91% |
| Fake News Detection | TF-IDF + PAC | WELFake | 90-93% |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/news-classifier.git
cd news-classifier

# Create and activate virtual environment
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLP data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm

# Train the models
python app/train_models.py

# Launch the Streamlit application
streamlit run app/streamlit_app.py
```

## Docker Development Setup (Cross-Machine)

If you are collaborating or developing this project on another machine, Docker Compose provides an isolated environment with hot-reloading already configured. The project's `.gitignore` and `.dockerignore` files are properly set up to prevent host machine dependencies (like `venv/` or `__pycache__/`) from conflicting with the container.

```bash
# 1. Clone the repository on the new machine
git clone https://github.com/your-username/news-classifier.git
cd news-classifier

# 2. Build and start the development container in detached mode
docker-compose up -d --build

# 3. Follow the logs to see the app running (Ctrl+C to exit logs)
docker-compose logs -f app

# The application is now accessible at http://localhost:8501
# Any changes you make to the local source code will automatically trigger a reload in the browser.

# 4. Stop the container when you are done
docker-compose down
```

## Project Structure

```text
news-classifier/
├── app/                  # Web app and training scripts
├── data/                 # Raw datasets and samples
├── notebooks/            # EDA and experimental notebooks
├── reports/              # Generated figures and metrics
├── saved_models/         # Serialized ML model artifacts (.joblib)
├── src/                  # Core ML backend (Models, Pipeline, OCR)
├── tests/                # Automated unit tests
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Datasets
* **AG News:** A large-scale news topic classification dataset from HuggingFace containing over 120,000 articles spanning multiple domains.
* **WELFake:** A comprehensive dataset from Kaggle comprising 72,000+ real and fake news articles used for training the authenticity verification model.

## Team
**COMSATS University Islamabad — Lahore Campus**  
BS Artificial Intelligence | Spring 2026  
Course: Introduction to Programming / AI Lab Project

* **Muhammad Ahad Bashir** (SP23-BAI-030) — Team Lead & Model Training
* **Muhammad Huzaifa Ali** (SP24-BAI-034) — NLP & Feature Engineering
* **Moiz Ul Rehman** (SP24-BAI-025) — Data & Evaluation

## License
This project is licensed under the MIT License.
