"""
pdf_extractor.py — Extracts text from PDF pages using OCR.

This module handles:
1. Opening a PDF file
2. Converting each page to an image (using pdf2image + Poppler)
3. Optionally, preprocessing images for better OCR quality
4. Running OCR (Optical Character Recognition) on each image
5. Running OCR (Optical Character Recognition) on each image
6. Returning the extracted text for each page
"""

from pdf2image import convert_from_path    # Converts PDF pages to PIL Images
from PIL import Image                      # Handles image objects in Python
import pytesseract                         # Python wrapper for Tesseract OCR
from pathlib import Path                   # Clean file path handling

from src.image_preprocessor import preprocess_image

from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from pathlib import Path

from src.image_preprocessor import preprocess_image


def select_better_text(raw_text, preprocessed_text):
    """
    Compares two OCR results and returns the one that is likely better.
    
    Heuristic: The version with more meaningful text (longer, after
    stripping whitespace) is usually more accurate, because lost text
    means the preprocessing was too aggressive.
    
    Parameters:
        raw_text (str): OCR result from the original image.
        preprocessed_text (str): OCR result from the preprocessed image.
        
    Returns:
        tuple: (best_text, source) where source is 'raw' or 'preprocessed'
    """
    # Strip whitespace and count meaningful characters
    # len() on stripped text tells us how much actual content we got
    raw_length = len(raw_text.strip())
    preprocessed_length = len(preprocessed_text.strip())
    
    # If preprocessing lost more than 20% of the text, it was too aggressive
    # Use the raw version instead
    # Why 20%? It's a reasonable threshold — small differences are fine,
    # but losing a fifth of the text means something went wrong
    if preprocessed_length < raw_length * 0.8:
        return raw_text, 'raw'
    
    # Otherwise, use the preprocessed version (it might be cleaner)
    return preprocessed_text, 'preprocessed'


def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts text from each page using OCR.
    
    For each page, runs OCR on both the original and a gently
    preprocessed version, then automatically selects the better result.
    
    Parameters:
        pdf_path (str or Path): The file path to the PDF document.
        
    Returns:
        list of dict: One dictionary per page, containing:
            - 'page_number': The page number (starting from 1)
            - 'text': The best OCR text (automatically selected)
            - 'text_source': Which version was selected ('raw' or 'preprocessed')
            - 'image': The original PIL Image object
    """
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    images = convert_from_path(str(pdf_path), dpi=200)
    
    results = []
    
    for i, original_image in enumerate(images, start=1):
        
        print(f"Processing page {i} of {len(images)}...")
        
        # Preprocess with gentle mode (contrast boost, not harsh threshold)
        preprocessed_image = preprocess_image(original_image, mode='gentle')
        
        # Run OCR on both versions
        raw_text = pytesseract.image_to_string(original_image, lang='eng')
        preprocessed_text = pytesseract.image_to_string(
            preprocessed_image, lang='eng'
        )
        
        # Automatically pick the better result
        best_text, source = select_better_text(raw_text, preprocessed_text)
        
        page_result = {
            'page_number': i,
            'text': best_text,
            'text_source': source,
            'image': original_image
        }
        
        results.append(page_result)
        print(f"  → Selected '{source}' OCR ({len(best_text.strip())} chars)")
    
    return results


if __name__ == "__main__":
    
    pdf_path = Path("data/input/anonymised_1.pdf")
    results = extract_text_from_pdf(pdf_path)
    
    for page in results:
        print(f"\n{'='*60}")
        print(f"PAGE {page['page_number']} (source: {page['text_source']})")
        print(f"{'='*60}")
        print(page['text'])
        print(f"\n[Text length: {len(page['text'])} characters]")

