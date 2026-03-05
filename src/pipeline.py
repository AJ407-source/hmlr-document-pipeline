"""
pipeline.py — The main document processing pipeline.

This is the orchestrator that connects all modules together:
1. PDF extraction (pdf_extractor.py)
2. Text cleaning (text_cleaner.py)
3. Page classification (page_classifier.py)
4. Name extraction (name_extractor.py)
5. Application number extraction (number_extractor.py)

It processes a PDF file from start to finish and saves the
results as a structured JSON file.
"""

import json
from pathlib import Path
from datetime import datetime

from src.pdf_extractor import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.page_classifier import classify_all_pages
from src.name_extractor import extract_names_from_pages
from src.number_extractor import extract_numbers_from_pages


def run_pipeline(pdf_path, output_path=None):
    """
    Runs the complete document processing pipeline on a PDF file.
    
    Parameters:
        pdf_path (str or Path): Path to the input PDF file.
        output_path (str or Path, optional): Path for the output JSON file.
            
    Returns:
        dict: The complete pipeline results.
    """
    pdf_path = Path(pdf_path)
    
    if output_path is None:
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pdf_path.stem}_results.json"
    else:
        output_path = Path(output_path)
    
    print(f"{'='*60}")
    print(f"HMLR Document Processing Pipeline")
    print(f"{'='*60}")
    print(f"Input:  {pdf_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # STEP 1: Extract text from the PDF
    print("STEP 1: Extracting text from PDF...")
    print("-" * 40)
    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages.\n")
    
    # STEP 2: Clean the extracted text
    print("STEP 2: Cleaning text...")
    print("-" * 40)
    for page in pages:
        original_length = len(page['text'])
        page['text'] = clean_text(page['text'])
        cleaned_length = len(page['text'])
        print(f"  Page {page['page_number']}: "
              f"{original_length} → {cleaned_length} chars "
              f"(removed {original_length - cleaned_length})")
    print()
    
    # STEP 3: Classify each page
    print("STEP 3: Classifying pages...")
    print("-" * 40)
    pages = classify_all_pages(pages)
    print()
    
    # STEP 4: Extract applicant names
    print("STEP 4: Extracting applicant names...")
    print("-" * 40)
    pages = extract_names_from_pages(pages)
    print()
    
    # STEP 5: Extract application numbers
    print("STEP 5: Extracting application numbers...")
    print("-" * 40)
    pages = extract_numbers_from_pages(pages)
    print()
    
    # STEP 6: Build the output structure and save
    print("STEP 6: Saving results...")
    print("-" * 40)
    
    results = {
        'metadata': {
            'source_file': str(pdf_path),
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_pages': len(pages),
            'pipeline_version': '1.0.0'
        },
        'pages': []
    }
    
    for page in pages:

        # Clean up applicant names — convert float32 to regular float
        # PyTorch returns float32 numbers, but JSON only understands
        # regular Python floats. float() converts between them.
        clean_names = []
        for name_info in page.get('applicant_names', []):
            clean_names.append({
                'name': name_info['name'],
                'type': name_info['type'],
                'confidence': float(name_info['confidence']),
                'method': name_info['method']
            })

        page_result = {
            'page_number': page['page_number'],
            'classification': {
                'category': page.get('category', 'unknown'),
                'confidence': float(round(page.get('confidence', 0.0), 4))
            },
            'applicant_names': clean_names,
            'application_numbers': [
                num['number'] for num in page.get('application_numbers', [])
            ],
            'text_preview': page['text'][:200] + '...'
        }
        
        
        results['pages'].append(page_result)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    
    # Print a summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE — SUMMARY")
    print(f"{'='*60}")
    
    for page in results['pages']:
        print(f"\nPage {page['page_number']}:")
        print(f"  Type:         {page['classification']['category']}")
        print(f"  Confidence:   {page['classification']['confidence']:.1%}")
        
        if page['applicant_names']:
            names = [n['name'] for n in page['applicant_names']]
            print(f"  Applicants:   {', '.join(names)}")
        else:
            print(f"  Applicants:   None found")
        
        if page['application_numbers']:
            print(f"  App Numbers:  {', '.join(page['application_numbers'])}")
        else:
            print(f"  App Numbers:  None found")
    
    return results


if __name__ == "__main__":
    
    pdf_path = Path("data/input/anonymised_1.pdf")
    results = run_pipeline(pdf_path)
    print(f"\nResults saved! Check the JSON file in data/output/")

