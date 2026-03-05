"""
text_cleaner.py — Cleans raw OCR text to improve downstream processing.

This module applies conservative text cleaning:
1. Removes garbage lines (non-text artifacts from OCR)
2. Normalises whitespace (removes excessive blank lines and spaces)
3. Fixes common OCR character substitution errors

The cleaning is intentionally conservative — we prefer to keep slightly
messy text rather than risk removing real content.
"""

import re      # Regular Expressions — Python's built-in pattern matching tool
               # 're' lets us search for patterns in text and replace them
               #
               # Think of regex like a "find and replace" tool on steroids.
               # Instead of searching for exact words, you can search for
               # PATTERNS like "any digit followed by a slash followed by
               # more digits"


def remove_garbage_lines(text):
    """
    Removes lines that are mostly non-alphabetic characters.
    
    OCR often reads table borders, decorative lines, and page artifacts
    as strings of symbols like "—-—- ||||| === @@@@". These lines
    contain very few real letters and can be safely removed.
    
    How it works:
        For each line, we count how many characters are actual letters
        (a-z, A-Z). If less than 30% of the characters are letters,
        we consider the line "garbage" and remove it.
    
    Why 30%?
        - A normal English sentence is ~80-90% letters
        - A line like "Application No. 02/80/1609" is ~60% letters
        - A garbage line like "—-— ||| === @@@" is ~0-5% letters
        - 30% is a safe cutoff that keeps real text but removes junk
    
    Parameters:
        text (str): The raw OCR text.
        
    Returns:
        str: Text with garbage lines removed.
    """
    # Split the text into individual lines
    lines = text.split('\n')
    
    # This list will hold only the "good" lines
    clean_lines = []
    
    for line in lines:
        # Skip completely empty lines for now (we'll handle spacing later)
        # strip() removes leading/trailing whitespace
        # If there's nothing left after stripping, it's a blank line
        if not line.strip():
            clean_lines.append('')
            continue
        
        # Count how many characters in this line are actual letters
        # We use a "list comprehension" — a compact way to filter a list
        #
        # [char for char in line if char.isalpha()]
        #   → This creates a list of only the letters in the line
        #
        # Example: "App No. 02/80" → ['A', 'p', 'p', 'N', 'o'] → 5 letters
        letter_count = len([char for char in line if char.isalpha()])
        
        # Total characters (excluding spaces — we don't want spaces
        # to dilute our percentage calculation)
        total_chars = len(line.strip())
        
        # Calculate the percentage of letters
        # We avoid division by zero with max(total_chars, 1)
        # max() returns whichever number is larger
        # So if total_chars is 0, we use 1 instead (can't divide by 0)
        letter_ratio = letter_count / max(total_chars, 1)
        
        # Keep the line only if at least 30% of characters are letters
        if letter_ratio >= 0.30:
            clean_lines.append(line)
    
    # Join the good lines back into a single string
    # '\n' means "put a newline character between each line"
    return '\n'.join(clean_lines)


def normalise_whitespace(text):
    """
    Cleans up excessive whitespace in the text.
    
    Fixes three problems:
    1. Multiple blank lines in a row → single blank line
    2. Multiple spaces in a row → single space
    3. Leading/trailing whitespace on each line → removed
    
    Parameters:
        text (str): The text to clean.
        
    Returns:
        str: Text with normalised whitespace.
    """
    # STEP 1: Replace multiple blank lines with a single blank line
    #
    # The regex pattern: r'\n\s*\n'
    #   \n    = a newline character
    #   \s*   = zero or more whitespace characters (spaces, tabs)
    #   \n    = another newline character
    #
    # So this matches: newline, optional whitespace, newline
    # Which is essentially "a blank line" (possibly with spaces on it)
    #
    # We replace 2+ blank lines with exactly 2 newlines (= one blank line)
    #
    # The '+' at the end means "one or more occurrences"
    # So \n\s*\n+ matches any number of consecutive blank lines
    text = re.sub(r'(\n\s*){2,}', '\n\n', text)
    
    # STEP 2: Replace multiple spaces with a single space
    #
    # r' +' matches one or more space characters
    # We replace with a single space
    # This turns "Application    No." into "Application No."
    text = re.sub(r' +', ' ', text)
    
    # STEP 3: Strip whitespace from each line
    # Split into lines, strip each one, rejoin
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # STEP 4: Strip leading/trailing whitespace from the whole text
    text = text.strip()
    
    return text


def fix_common_ocr_errors(text):
    """
    Fixes character substitutions that OCR commonly makes.
    
    OCR engines sometimes confuse similar-looking characters.
    We fix only SAFE substitutions — ones where the wrong character
    would never appear in normal text at that position.
    
    IMPORTANT: We are very conservative here. We only fix errors
    that appear at the START of known words, where we can be
    confident the substitution is wrong.
    
    We do NOT do global find-and-replace (like changing all '0' to 'O')
    because that would corrupt application numbers and dates.
    
    Parameters:
        text (str): The text with OCR errors.
        
    Returns:
        str: Text with common OCR errors fixed.
    """
    # Each tuple is (wrong_pattern, correct_replacement)
    #
    # We use \b for "word boundary" — this ensures we only match
    # at the START of a word, not in the middle
    #
    # Example: \b-ondon matches "-ondon" at the start of a word
    #          but would NOT match "p-ondon" (not at word boundary)
    
    corrections = [
        # Capital letters misread as symbols at start of words
        (r'\b-ondon\b', 'London'),          # L misread as -
        (r'\b=ast\b', 'East'),              # E misread as =
        (r'\b=ax\b', 'Fax'),                # F misread as =
        (r'\brelephone\b', 'Telephone'),    # T misread as r
        (r'\bfown\b', 'Town'),              # T misread as f
        (r'\bzast\b', 'East'),              # E misread as z
        
        # Common word-level OCR errors
        (r'\bhonses\b', 'houses'),          # u misread as n
        (r'\bsegistration\b', 'registration'),  # r misread as s
        (r'\btise\b', 'rise'),              # r misread as t
        (r'\bMre\b', 'Mrs'),               # s misread as e
    ]
    
    for wrong_pattern, correct_text in corrections:
        # re.IGNORECASE makes the match case-insensitive
        # But we replace with the exact case we specify
        text = re.sub(wrong_pattern, correct_text, text, flags=re.IGNORECASE)
    
    return text


def clean_text(text):
    """
    Applies all text cleaning steps in order.
    
    Pipeline: Remove garbage → Normalise whitespace → Fix OCR errors
    
    Parameters:
        text (str): The raw OCR text.
        
    Returns:
        str: The cleaned text, ready for classification and extraction.
    """
    text = remove_garbage_lines(text)
    text = normalise_whitespace(text)
    text = fix_common_ocr_errors(text)
    
    return text


if __name__ == "__main__":
    
    # Test the cleaner on our actual OCR output
    from pathlib import Path
    from pdf2image import convert_from_path
    import pytesseract
    from src.image_preprocessor import preprocess_image
    
    pdf_path = Path("data/input/anonymised_1.pdf")
    images = convert_from_path(str(pdf_path), dpi=200)
    
    # Process each page
    for i, original_image in enumerate(images, start=1):
        
        # Get OCR text (using gentle preprocessing)
        preprocessed = preprocess_image(original_image, mode='gentle')
        raw_text = pytesseract.image_to_string(preprocessed, lang='eng')
        
        # Clean the text
        cleaned_text = clean_text(raw_text)
        
        print(f"\n{'='*60}")
        print(f"PAGE {i}")
        print(f"{'='*60}")
        print(f"BEFORE cleaning: {len(raw_text)} chars")
        print(f"AFTER cleaning:  {len(cleaned_text)} chars")
        print(f"Removed: {len(raw_text) - len(cleaned_text)} chars")
        print(f"{'='*60}")
        print(cleaned_text[:500])
        print("...")

