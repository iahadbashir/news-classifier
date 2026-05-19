"""OCR reader module to extract text from images for the news classifier."""

import os
import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError


class OCRReader:
    """
    A class to extract text from images using OpenCV and Tesseract OCR.
    
    Includes an image preprocessing pipeline (grayscale, thresholding, denoising)
    to improve OCR accuracy on noisy or poorly lit images.
    """

    def __init__(self):
        """Initialize the OCRReader."""
        pass

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to an image to improve OCR accuracy.
        
        Converts the image to grayscale, denoises using Gaussian Blur,
        and applies adaptive thresholding.
        
        Args:
            img (np.ndarray): The raw image loaded via OpenCV.
            
        Returns:
            np.ndarray: The preprocessed image ready for OCR.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to denoise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh

    def _clean_ocr_text(self, text: str) -> str:
        """
        Strip excessive whitespace and empty lines from OCR output.
        
        Args:
            text (str): Raw text extracted by Tesseract.
            
        Returns:
            str: Cleaned text string.
        """
        # Split into lines, strip whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        # Keep only non-empty lines
        non_empty_lines = [line for line in lines if line]
        # Join with a single newline
        return "\n".join(non_empty_lines).strip()

    def read_from_file(self, image_path: str) -> str:
        """
        Open an image file from disk, preprocess it, and extract text.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            str: The extracted and cleaned text.
            
        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the file exists but cannot be read as an image.
            RuntimeError: If Tesseract OCR is not installed.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")
            
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}. File may be corrupt or unsupported.")

        processed_img = self._preprocess_image(img)

        try:
            raw_text = pytesseract.image_to_string(processed_img)
        except TesseractNotFoundError:
            raise RuntimeError("Tesseract OCR is not installed. Download from: https://github.com/UB-Mannheim/tesseract/wiki")

        return self._clean_ocr_text(raw_text)

    def read_from_bytes(self, image_bytes: bytes) -> str:
        """
        Process raw image bytes (e.g., from Streamlit file_uploader), preprocess, and extract text.
        
        Args:
            image_bytes (bytes): The raw image byte data.
            
        Returns:
            str: The extracted and cleaned text.
            
        Raises:
            ValueError: If the image bytes are invalid or cannot be decoded.
            RuntimeError: If Tesseract OCR is not installed.
        """
        # Convert bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes. The uploaded file may be invalid or corrupt.")

        processed_img = self._preprocess_image(img)

        try:
            raw_text = pytesseract.image_to_string(processed_img)
        except TesseractNotFoundError:
            raise RuntimeError("Tesseract OCR is not installed. Download from: https://github.com/UB-Mannheim/tesseract/wiki")

        return self._clean_ocr_text(raw_text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/vision/ocr_reader.py <path_to_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    print(f"Reading text from: {img_path}...\n")
    
    reader = OCRReader()
    try:
        extracted_text = reader.read_from_file(img_path)
        print("--- EXTRACTED TEXT ---")
        print(extracted_text)
        print("----------------------")
    except Exception as e:
        print(f"Error: {e}")
