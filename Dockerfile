FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Download the Spacy English model
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
