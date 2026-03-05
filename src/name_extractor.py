"""
name_extractor.py — Extracts applicant names from OCR text using NER.

This module uses the dslim/bert-base-NER transformer model to identify
person names (PER) and organisation names (ORG) in the text.

It then applies heuristic filtering to find the most likely "applicant"
name, rather than returning every name found in the document.
"""

from transformers import pipeline


def load_ner_model():
    """
    Loads the Named Entity Recognition (NER) model.
    
    Downloads the model the first time (about 400 MB).
    After that, it's cached and loads quickly.
    
    Returns:
        transformers.Pipeline: The NER model, ready to use.
    """
    print("Loading NER model (this may take a moment)...")
    
    ner_model = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )
    
    print("NER model loaded successfully!")
    
    return ner_model


def extract_entities(ner_model, text):
    """
    Runs NER on text and returns all person and organisation entities found.
    
    Parameters:
        ner_model: The loaded NER model (from load_ner_model()).
        text (str): The cleaned OCR text from one page.
        
    Returns:
        list of dict: Each dict contains:
            - 'name': The extracted name text
            - 'type': Either 'PER' (person) or 'ORG' (organisation)
            - 'confidence': How confident the model is (0.0 to 1.0)
    """
    if len(text.strip()) < 20:
        return []
    
    text_to_process = text[:1500]
    raw_entities = ner_model(text_to_process)
    
    relevant_entities = []
    
    for entity in raw_entities:
        
        entity_type = entity['entity_group']
        
        if entity_type in ['PER', 'ORG']:
            
            name = entity['word'].strip()
            
            if len(name) < 2:
                continue
            
            relevant_entities.append({
                'name': name,
                'type': entity_type,
                'confidence': round(entity['score'], 4)
            })
    
    return relevant_entities


def is_valid_name(name):
    """
    Checks if an extracted name looks like a real person or company name.
    
    Filters out junk that the NER model mistakenly identified as names.
    
    Parameters:
        name (str): The extracted name text.
        
    Returns:
        bool: True if it looks like a real name, False if it's junk.
    """
    # Rule 1: Must be at least 3 characters
    # Real names are at least 3 chars (e.g., "Lee", "Ali", "Fox")
    # This removes: "OR", "Po", "and"
    if len(name.strip()) < 3:
        return False
    
    # Rule 2: Must contain at least one uppercase letter
    # Real names almost always start with a capital letter
    # This removes junk like "end", "and"
    #
    # any() checks if ANY character passes the test
    # char.isupper() checks if a character is uppercase (A-Z)
    has_uppercase = any(char.isupper() for char in name)
    if not has_uppercase:
        return False
    
    # Rule 3: Must not start with ## (BERT subword tokens)
    # These are internal model artifacts, not real text
    # This removes: "##VO", "##VAL H"
    if name.startswith('##'):
        return False
    
    # Rule 4: Must contain at least one letter that is NOT uppercase
    # This filters out abbreviations like "APPR", "OR"
    # Real names have a mix: "Dale", "Stephens", "Company"
    # Pure uppercase short strings are usually abbreviations
    #
    # But we allow fully uppercase if longer than 6 chars
    # because some names in old documents ARE in all caps
    # like "MR J M DOE"
    if name.isupper() and len(name) < 6:
        return False
    
    # Rule 5: Must have more letters than non-letters
    # This filters out things like "P/96" or "02/80"
    # that might accidentally be tagged as entities
    letter_count = sum(1 for char in name if char.isalpha())
    total_count = len(name.strip())
    
    if total_count > 0 and letter_count / total_count < 0.5:
        return False
    
    # Rule 6: Must not be a common English word
    # NER sometimes tags ordinary words as names
    # These are words that appeared in our junk results
    common_words = [
        'and', 'the', 'for', 'not', 'end', 'act',
        'planning', 'borough', 'county', 'town',
        'development', 'country', 'district'
    ]
    
    if name.lower().strip() in common_words:
        return False
    
    # Passed all checks — looks like a real name!
    return True


def find_applicant_names(ner_model, text):
    """
    Finds the most likely applicant name(s) from a page of text.
    
    Strategy:
    1. Run NER to find all person and organisation names
    2. Filter out junk entities using is_valid_name()
    3. Look for names that appear near "Applicant" keywords
    4. Filter out names that are likely authorities or officers
    
    Parameters:
        ner_model: The loaded NER model.
        text (str): The cleaned OCR text from one page.
        
    Returns:
        list of dict: Likely applicant names, each containing:
            - 'name': The applicant's name
            - 'type': 'PER' or 'ORG'
            - 'confidence': Model confidence score
            - 'method': How we identified this as the applicant
    """
    all_entities = extract_entities(ner_model, text)
    
    if not all_entities:
        return []
    
    text_lower = text.lower()
    
    applicant_keywords = [
        'applicant',
        'granted to',
        'approval granted to',
        'permission granted to',
        'applied by',
        'submitted by',
        'on behalf of'
    ]
    
    exclude_names = [
        'london borough of newham',
        'north devon district council',
        'newham',
        'wyre',
        'thornton',
        'council',
        'secretary of state',
        'department of the environment',
        'town and country',
        'town and county',
        'and country planning',
        'and county planning',
        'development pro',
        'development proced',
        'anne',
        'annexe'
    ]
    
    applicant_candidates = []
    
    for entity in all_entities:
        
        name = entity['name']
        name_lower = name.lower()
        
        # NEW: Check if the name passes our validity checks
        if not is_valid_name(name):
            continue
        
        # Check against exclude list
        is_excluded = any(
            excluded in name_lower for excluded in exclude_names
        )
        
        if is_excluded:
            continue
        
        # Check proximity to applicant keywords
        found_near_keyword = False
        
        for keyword in applicant_keywords:
            
            keyword_pos = text_lower.find(keyword)
            
            if keyword_pos == -1:
                continue
            
            name_pos = text_lower.find(name_lower)
            
            if name_pos == -1:
                continue
            
            distance = name_pos - keyword_pos
            
            if 0 < distance < 200:
                found_near_keyword = True
                break
        
        if found_near_keyword:
            method = 'keyword_proximity'
        else:
            method = 'ner_only'
        
        applicant_candidates.append({
            'name': name,
            'type': entity['type'],
            'confidence': entity['confidence'],
            'method': method
        })
    
    # Sort: keyword matches first, then by confidence
    applicant_candidates.sort(
        key=lambda x: (x['method'] != 'keyword_proximity', -x['confidence'])
    )

    has_keyword_matches = any(
        c['method'] == 'keyword_proximity' for c in applicant_candidates
    )

    if has_keyword_matches:
        applicant_candidates = [
            c for c in applicant_candidates
            if c['method'] == 'keyword_proximity'
        ]
    
    return applicant_candidates


def extract_names_from_pages(pages):
    """
    Extracts applicant names from every page.
    
    Loads the NER model once, then processes each page.
    
    Parameters:
        pages (list of dict): Pages with 'text' and 'page_number' keys.
        
    Returns:
        list of dict: The same pages, with 'applicant_names' added.
    """
    ner_model = load_ner_model()
    
    for page in pages:
        
        print(f"Extracting names from page {page['page_number']}...")
        
        applicants = find_applicant_names(ner_model, page['text'])
        
        page['applicant_names'] = applicants
        
        if applicants:
            for applicant in applicants:
                print(f"  → {applicant['name']} "
                      f"({applicant['type']}, "
                      f"confidence: {applicant['confidence']:.1%}, "
                      f"method: {applicant['method']})")
        else:
            print("  → No applicant names found")
    
    return pages


if __name__ == "__main__":
    
    from pathlib import Path
    from pdf2image import convert_from_path
    import pytesseract
    from src.image_preprocessor import preprocess_image
    from src.text_cleaner import clean_text
    
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
    
    pages = extract_names_from_pages(pages)
    
    print(f"\n{'='*60}")
    print("NAME EXTRACTION RESULTS")
    print(f"{'='*60}")
    
    for page in pages:
        print(f"\nPage {page['page_number']}:")
        if page['applicant_names']:
            for name_info in page['applicant_names']:
                print(f"  Name:       {name_info['name']}")
                print(f"  Type:       {name_info['type']}")
                print(f"  Confidence: {name_info['confidence']:.1%}")
                print(f"  Method:     {name_info['method']}")
                print()
        else:
            print("  No applicant names found")

