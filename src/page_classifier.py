"""
page_classifier.py — Classifies document pages using a zero-shot transformer.

This module uses Facebook's BART-MNLI model to classify each page
into one of three planning document categories:
- approval_notice: A formal notice approving a planning application
- planning_register: A register or log of planning charges/entries
- general_permission: A general grant of permission or legal terms

The model is "zero-shot" — it needs no training examples, just the
category descriptions.
"""

from transformers import pipeline


def load_classifier():
    """
    Loads the zero-shot classification model.
    
    This downloads the model the FIRST time you run it (about 1.5 GB).
    After that, it's cached on your machine and loads much faster.
    
    Returns:
        transformers.Pipeline: The classifier, ready to use.
    """
    print("Loading classification model (this may take a moment)...")
    
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    
    print("Model loaded successfully!")
    
    return classifier


# Category labels — descriptive phrases the model can understand
CANDIDATE_LABELS = [
    "planning approval notice or decision",
    "planning charges register or record",
    "general grant of planning permission or legal terms"
]

# Maps long labels to short category names for clean output
LABEL_TO_CATEGORY = {
    "planning approval notice or decision": "approval_notice",
    "planning charges register or record": "planning_register",
    "general grant of planning permission or legal terms": "general_permission"
}


def classify_page(classifier, text):
    """
    Classifies a single page of text into a document category.
    
    Parameters:
        classifier: The loaded zero-shot classifier (from load_classifier()).
        text (str): The cleaned OCR text from one page.
        
    Returns:
        dict: Classification result containing:
            - 'category': The short category name (e.g., 'approval_notice')
            - 'label': The full descriptive label the model matched
            - 'confidence': How confident the model is (0.0 to 1.0)
            - 'all_scores': Confidence scores for all categories
    """
    if len(text.strip()) < 50:
        return {
            'category': 'unknown',
            'label': 'insufficient text',
            'confidence': 0.0,
            'all_scores': {}
        }
    
    result = classifier(
        text[:1000],
        candidate_labels=CANDIDATE_LABELS,
        multi_label=False
    )
    
    best_label = result['labels'][0]
    best_score = result['scores'][0]
    category = LABEL_TO_CATEGORY.get(best_label, 'unknown')
    
    all_scores = dict(zip(result['labels'], result['scores']))
    
    return {
        'category': category,
        'label': best_label,
        'confidence': best_score,
        'all_scores': all_scores
    }


def classify_all_pages(pages):
    """
    Classifies every page in a list of extracted pages.
    
    Loads the model once, then classifies each page.
    
    Parameters:
        pages (list of dict): The output from extract_text_from_pdf().
            Each dict must have 'text' and 'page_number' keys.
            
    Returns:
        list of dict: The same page dicts, but with classification added:
            - 'category': The document type
            - 'confidence': How confident the model is
            - 'all_scores': Scores for all categories
    """
    classifier = load_classifier()
    
    for page in pages:
        
        print(f"Classifying page {page['page_number']}...")
        
        classification = classify_page(classifier, page['text'])
        
        page['category'] = classification['category']
        page['confidence'] = classification['confidence']
        page['all_scores'] = classification['all_scores']
        
        print(f"  → {classification['category']} "
              f"(confidence: {classification['confidence']:.1%})")
    
    return pages


if __name__ == "__main__":
    
    from pathlib import Path
    from pdf2image import convert_from_path
    import pytesseract
    from src.image_preprocessor import preprocess_image
    from src.text_cleaner import clean_text
    
    # Step 1: Get OCR text from the PDF
    pdf_path = Path("data/input/anonymised_1.pdf")
    images = convert_from_path(str(pdf_path), dpi=200)
    
    pages = []
    for i, image in enumerate(images, start=1):
        preprocessed = preprocess_image(image, mode='gentle')
        text = pytesseract.image_to_string(preprocessed, lang='eng')
        cleaned = clean_text(text)
        pages.append({
            'page_number': i,
            'text': cleaned,
            'image': image
        })
    
    # Step 2: Classify each page
    pages = classify_all_pages(pages)
    
    # Step 3: Print the results
    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    
    for page in pages:
        print(f"\nPage {page['page_number']}:")
        print(f"  Category:   {page['category']}")
        print(f"  Confidence: {page['confidence']:.1%}")
        print(f"  All scores:")
        for label, score in page['all_scores'].items():
            short_name = LABEL_TO_CATEGORY.get(label, label)
            print(f"    {short_name}: {score:.1%}")

