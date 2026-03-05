"""
pdf_extractor.py — Extracts text from PDF pages using OCR.

This module handles:
1. Opening a PDF file
2. Converting each page to an image
3. Running OCR (Optical Character Recognition) on each image
4. Returning the extracted text for each page
"""

import fitz                    # PyMuPDF — opens PDFs and converts pages to images
from PIL import Image          # Pillow — handles image objects in Python
import pytesseract             # Python wrapper for Tesseract OCR engine
import io                     # Built-in Python module for handling byte streams
from pathlib import Path       # Built-in Python module for clean file path handling


def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts text from each page using OCR.
    
    Parameters:
        pdf_path (str or Path): The file path to the PDF document.
        
    Returns:
        list of dict: One dictionary per page, containing:
            - 'page_number': The page number (starting from 1)
            - 'text': The raw text extracted by OCR
            - 'image': The PIL Image object (kept for later preprocessing)
    """
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(str(pdf_path))
    
    results = []
    
    for i, page in enumerate(doc, start=1):
        
        print(f"Processing page {i} of {len(doc)}...")
        
        # Convert PDF page to image at 2x resolution for better OCR
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        
        # Convert pixmap bytes to a PIL Image object
        image_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run OCR on the image
        text = pytesseract.image_to_string(image, lang='eng')
        
        # Store results for this page
        page_result = {
            'page_number': i,
            'text': text,
            'image': image
        }
        
        results.append(page_result)
    
    doc.close()
    
    return results


if __name__ == "__main__":
    
    pdf_path = Path("data/input/anonymised_1.pdf")
    
    results = extract_text_from_pdf(pdf_path)
    
    for page in results:
        print(f"\n{'='*60}")
        print(f"PAGE {page['page_number']}")
        print(f"{'='*60}")
        print(page['text'])
        print(f"\n[Text length: {len(page['text'])} characters]")

