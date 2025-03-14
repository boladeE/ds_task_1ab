from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    """
    Extracts text from an image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Extracted text.
    """
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(image)
    
    return extracted_text

 