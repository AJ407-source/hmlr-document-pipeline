"""
number_extractor.py — Extracts planning application numbers using Regex.

Application numbers in planning documents follow predictable formats:
- Format 1: XX/XX/XXXX  (e.g., 02/80/1609)
- Format 2: X/XX/XXXX   (e.g., P/00/0759)

This module uses Regular Expressions to find these patterns in OCR text.
Regex is more appropriate than AI for this task because the patterns
are strict and predictable.

A date filter removes matches that are actually dates (e.g., 17/07/2000)
rather than application numbers.
"""

import re


APPLICATION_PATTERNS = [
    r'\d{2}\/\d{2}\/\d{3,4}',
    r'[A-Z]\/\d{2}\/\d{3,4}',
]


def is_likely_date(text, match_text, match_position):
    """
    Checks if a regex match is probably a date rather than an application number.
    
    Uses two methods:
    1. Check if the middle number is a valid month (01-12) AND the last
       number is a plausible year (1900-2099)
    2. Check if date-related words appear nearby in the text
    
    Parameters:
        text (str): The full page text we're searching in.
        match_text (str): The matched string (e.g., '17/07/2000').
        match_position (int): Where in the text this match starts.
        
    Returns:
        bool: True if this looks like a date, False if it looks like
              an application number.
    """
    # Split the match into its three parts
    # '17/07/2000'.split('/') gives us ['17', '07', '2000']
    #
    # split('/') chops a string wherever it finds a '/'
    # Like cutting a ribbon at every knot
    parts = match_text.split('/')
    
    # We only check Format 1 matches (all digits)
    # Format 2 matches (starting with a letter like P/00/0759)
    # are NEVER dates, so we skip this check for them
    if not parts[0].isdigit():
        return False
    
    # Convert the parts to numbers so we can check their values
    # int() converts a string to an integer (number)
    # int('07') → 7
    # int('2000') → 2000
    first = int(parts[0])
    middle = int(parts[1])
    last = int(parts[2])
    
    # METHOD 1: Check if the numbers look like a date
    #
    # A date has:
    #   - First number (day): 1-31
    #   - Middle number (month): 1-12
    #   - Last number (year): 1900-2099
    #
    # An application number typically has:
    #   - Middle number > 12 (it's a year like 80, 81)
    #   - OR last number < 1900 (it's a sequence number like 1609)
    is_valid_day = 1 <= first <= 31
    is_valid_month = 1 <= middle <= 12
    is_valid_year = 1900 <= last <= 2099
    
    looks_like_date = is_valid_day and is_valid_month and is_valid_year
    
    # METHOD 2: Check for date-related words nearby
    #
    # We look at the 50 characters BEFORE the match in the text
    # If words like "Date" or "dated" appear there, it's probably a date
    #
    # Why 50 characters? Because "Date of Application:" is about 20 chars,
    # and there might be some spacing. 50 is generous but safe.
    #
    # max(0, match_position - 50) ensures we don't go before the start
    # of the text (negative positions would cause errors)
    context_start = max(0, match_position - 50)
    context_before = text[context_start:match_position].lower()
    
    # These words suggest the number after them is a date
    date_keywords = ['date', 'dated', 'received', 'on the']
    
    # any() checks: "Do ANY of these keywords appear in the context?"
    has_date_keyword = any(
        keyword in context_before for keyword in date_keywords
    )
    
    # If it BOTH looks like a date AND has a date keyword nearby,
    # we're very confident it's a date
    # If it looks like a date but has no keyword, we still filter it
    # because the number pattern (day/month/year) is strong evidence
    if looks_like_date:
        return True
    
    if has_date_keyword:
        return True
    
    return False


def extract_application_numbers(text):
    """
    Finds all planning application numbers in the given text.
    
    Tries each regex pattern against the text and collects
    all matches. Dates are filtered out. Duplicates are removed.
    
    Parameters:
        text (str): The cleaned OCR text from one page.
        
    Returns:
        list of dict: Each dict contains:
            - 'number': The application number found
            - 'pattern': Which regex pattern matched it
    """
    found_numbers = []
    seen = set()
    
    for pattern in APPLICATION_PATTERNS:
        
        # CHANGED: Use re.finditer() instead of re.findall()
        #
        # re.findall() only gives us the matched TEXT
        # re.finditer() gives us match OBJECTS that include:
        #   - The matched text (.group())
        #   - WHERE in the text it was found (.start())
        #
        # We need the position to check for nearby date keywords
        #
        # finditer returns an "iterator" — like a conveyor belt
        # that delivers matches one at a time
        for match in re.finditer(pattern, text):
            
            # .group() gives us the actual matched text
            match_text = match.group()
            
            # .start() gives us the character position where the match begins
            match_position = match.start()
            
            # Skip if we've already found this number
            if match_text in seen:
                continue
            
            # NEW: Skip if this looks like a date
            if is_likely_date(text, match_text, match_position):
                continue
            
            seen.add(match_text)
            
            found_numbers.append({
                'number': match_text,
                'pattern': pattern
            })
    
    return found_numbers


def extract_numbers_from_pages(pages):
    """
    Extracts application numbers from every page.
    
    Parameters:
        pages (list of dict): Pages with 'text' and 'page_number' keys.
        
    Returns:
        list of dict: The same pages, with 'application_numbers' added.
    """
    for page in pages:
        
        print(f"Extracting application numbers from page {page['page_number']}...")
        
        numbers = extract_application_numbers(page['text'])
        
        page['application_numbers'] = numbers
        
        if numbers:
            for num in numbers:
                print(f"  → {num['number']}")
        else:
            print("  → No application numbers found")
    
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
    
    pages = extract_numbers_from_pages(pages)
    
    print(f"\n{'='*60}")
    print("APPLICATION NUMBER EXTRACTION RESULTS")
    print(f"{'='*60}")
    
    for page in pages:
        print(f"\nPage {page['page_number']}:")
        if page['application_numbers']:
            for num_info in page['application_numbers']:
                print(f"  Number:  {num_info['number']}")
                print(f"  Pattern: {num_info['pattern']}")
                print()
        else:
            print("  No application numbers found")


