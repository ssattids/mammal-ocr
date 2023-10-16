
# %%
import pytesseract
from PIL import Image

# # Set the path to the Tesseract executable (change this to the correct path on your system)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open an image using Pillow (PIL)
image = Image.open('./data/testocr.png')

# Use pytesseract to extract text from the image
text = pytesseract.image_to_string(image)

# Print the extracted text
print(text)
# %%
