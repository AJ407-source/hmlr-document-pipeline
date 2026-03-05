"""
image_preprocessor.py — Preprocesses scanned page images to improve OCR quality.

This module applies image cleaning techniques using Pillow (PIL):
1. Convert to greyscale
2. Remove noise (denoise)
3. Apply binary thresholding
4. Sharpen the image

These steps make text clearer and help Tesseract produce more accurate results.

Note: We use Pillow instead of OpenCV because OpenCV triggers an OpenSSL
FIPS self-test failure on systems with FIPS mode enabled (e.g., Rocky Linux).
Pillow provides all the image processing we need without this issue.
"""

from PIL import Image              # Core image handling
from PIL import ImageFilter         # Provides filters: blur, sharpen, denoise
from PIL import ImageEnhance       # For adjusting contrast/brightness
from pathlib import Path           # Clean file path handling


def convert_to_greyscale(pil_image):
    """
    Converts a colour image to greyscale (shades of grey only).
    
    Parameters:
        pil_image (PIL.Image): The colour image.
        
    Returns:
        PIL.Image: The image in greyscale.
    """
    greyscale = pil_image.convert('L')
    return greyscale


def remove_noise(pil_image):
    """
    Removes small speckles and noise from the image.
    
    Parameters:
        pil_image (PIL.Image): The greyscale image.
        
    Returns:
        PIL.Image: The denoised image.
    """
    denoised = pil_image.filter(ImageFilter.MedianFilter(size=3))
    return denoised


def enhance_contrast(pil_image, factor=1.5):
    """
    Increases the contrast of the image to make text stand out more.
    
    Instead of the harsh binary threshold (which lost text), this
    gently boosts the difference between dark and light areas.
    
    Parameters:
        pil_image (PIL.Image): The greyscale image.
        factor (float): How much to boost contrast.
            1.0 = no change
            1.5 = 50% more contrast (good default)
            2.0 = double the contrast (aggressive)
        
    Returns:
        PIL.Image: The contrast-enhanced image.
    """
    # ImageEnhance.Contrast creates a contrast adjuster for the image
    # .enhance(factor) applies the adjustment
    #
    # Think of it like the "contrast" slider on your phone's photo editor
    # We're sliding it up a bit — not all the way to maximum
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(factor)
    return enhanced


def sharpen_image(pil_image):
    """
    Sharpens the image, making text edges crisper.
    
    Parameters:
        pil_image (PIL.Image): The image to sharpen.
        
    Returns:
        PIL.Image: The sharpened image.
    """
    sharpened = pil_image.filter(ImageFilter.SHARPEN)
    return sharpened


def apply_threshold(pil_image, cutoff=128):
    """
    Converts the image to pure black-and-white (binary).
    
    WARNING: This can cause text loss on faded/scanned documents.
    Use enhance_contrast() as a gentler alternative.
    
    Parameters:
        pil_image (PIL.Image): The greyscale image.
        cutoff (int): Brightness cutoff (0-255). Default 128.
        
    Returns:
        PIL.Image: The binary (black and white) image.
    """
    binary = pil_image.point(lambda pixel: 255 if pixel > cutoff else 0)
    return binary


def preprocess_image(pil_image, mode='gentle'):
    """
    Applies preprocessing steps to a PIL Image for better OCR.
    
    Two modes available:
        'gentle': Greyscale → Denoise → Contrast boost → Sharpen
                  (preserves most text, good for faded scans)
        
        'full':   Greyscale → Denoise → Binary threshold → Sharpen
                  (aggressive, may lose faded text)
    
    Parameters:
        pil_image (PIL.Image): The original page image from the PDF.
        mode (str): 'gentle' or 'full'. Default is 'gentle'.
        
    Returns:
        PIL.Image: The preprocessed image, ready for OCR.
    """
    greyscale = convert_to_greyscale(pil_image)
    denoised = remove_noise(greyscale)
    
    if mode == 'gentle':
        # Boost contrast instead of harsh thresholding
        # This makes dark text darker and light background lighter
        # WITHOUT destroying text that falls near the boundary
        enhanced = enhance_contrast(denoised, factor=1.5)
        sharpened = sharpen_image(enhanced)
    elif mode == 'full':
        # Aggressive binary thresholding (kept for completeness)
        binary = apply_threshold(denoised)
        sharpened = sharpen_image(binary)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'gentle' or 'full'.")
    
    return sharpened


if __name__ == "__main__":
    
    from pdf2image import convert_from_path
    import pytesseract
    
    pdf_path = Path("data/input/anonymised_1.pdf")
    images = convert_from_path(str(pdf_path), dpi=200)
    
    # Test all three approaches on page 1
    original_image = images[0]
    gentle_image = preprocess_image(original_image, mode='gentle')
    full_image = preprocess_image(original_image, mode='full')
    
    # Create output directory
    Path("data/output").mkdir(parents=True, exist_ok=True)
    
    # Save all three for visual comparison
    original_image.save("data/output/page1_original.png")
    gentle_image.save("data/output/page1_gentle.png")
    full_image.save("data/output/page1_full.png")
    
    # Run OCR on all three
    original_text = pytesseract.image_to_string(original_image, lang='eng')
    gentle_text = pytesseract.image_to_string(gentle_image, lang='eng')
    full_text = pytesseract.image_to_string(full_image, lang='eng')
    
    print(f"{'='*60}")
    print(f"ORIGINAL OCR ({len(original_text)} chars):")
    print(f"{'='*60}")
    print(original_text[:400])
    
    print(f"\n{'='*60}")
    print(f"GENTLE PREPROCESSING ({len(gentle_text)} chars):")
    print(f"{'='*60}")
    print(gentle_text[:400])
    
    print(f"\n{'='*60}")
    print(f"FULL PREPROCESSING ({len(full_text)} chars):")
    print(f"{'='*60}")
    print(full_text[:400])

