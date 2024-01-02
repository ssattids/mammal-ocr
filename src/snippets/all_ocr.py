# %%
import boto3
import io
import fitz
from PIL import Image, ImageDraw
import json
import sklearn.neighbors as skneighbors
import numpy as np
import pandas as pd
import json
import random
# %%
input_pdf = "/Users/salarsatti/projects/mammal-ocr/src/data/Travelers-RMD-MO-130-133-Forms_removed.pdf"
# input_pdf = "/Users/salarsatti/projects/mammal-ocr/src/data/ACORD_130.pdf"
dpi = 400
zoom = 1 # to increase the resolution
mat = fitz.Matrix(dpi/72, dpi/72)
# %%
# open pdf and loop over images
doc = fitz.open(input_pdf)
images_bytes = []
images = []
all_page_data = []
image_format = 'PNG'
for page_index in range(len(doc)):
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    image_bytes = pix.tobytes(image_format)
    # PIL open function requires a file like object
    image = Image.open(io.BytesIO(image_bytes))
    image_width, image_height = image.size
    # append to images array to be fed to asyn function
    images_bytes.append(image_bytes)

    all_page_data.append({
        "page_index": page_index,
        "page_number":page_index + 1,
        "image_height": image_height,
        "image_width": image_width,
        "image_bytes": image_bytes,
        "image_format": image_format,
        "dpi": dpi
    })
    images.append(image)
    image.save(f"./page_{page_index}.png")
    break
    # display(image)
# %%
import pytesseract
from pytesseract import Output
import cv2
img = cv2.imread('./page_0.png')

d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    if d['conf'][i] == -1:
        continue
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
# %%
import cv2
import pytesseract

filename = './page_0.png'

# read the image and get the dimensions
img = cv2.imread(filename)
h, w, _ = img.shape # assumes color image

# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img) # also include any config options you use

# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

# show annotated image and wait for keypress
cv2.imshow(filename, img)
cv2.waitKey(0)
# %%
# %%
import pytesseract
import cv2
from pytesseract import Output

img = cv2.imread("./page_0.jpg")
height = img.shape[0]
width = img.shape[1]

d = pytesseract.image_to_boxes(img, output_type=Output.DICT)
n_boxes = len(d['char'])
for i in range(n_boxes):
    (text,x1,y2,x2,y1) = (d['char'][i],d['left'][i],d['top'][i],d['right'][i],d['bottom'][i])
    cv2.rectangle(img, (x1,height-y1), (x2,height-y2) , (0,255,0), 2)
cv2.imshow('img',img)
cv2.waitKey(0)
# %%
import pytesseract
from PIL import Image

# Path to Tesseract executable (change this based on your installation)

# Path to your image
image_path = "./page_0.png"

# Open the image using PIL (Python Imaging Library)
img = Image.open(image_path)

# Use pytesseract to get the bounding boxes and text
boxes = pytesseract.image_to_boxes(img)
text = pytesseract.image_to_string(img)

# Printing the extracted text
print("Extracted Text:")
print(text)

# Printing the bounding boxes
print("\nBounding Boxes:")
for b in boxes.splitlines():
    b = b.split()
    # b[0] is the character, while b[1:5] are the coordinates of the box
    print(f"Character: {b[0]}, Bounding Box Coordinates: {', '.join(b[1:5])}")

# %%
import pytesseract
import cv2

# Path to Tesseract executable (change this based on your installation)

# Path to your image
image_path = "/Users/salarsatti/projects/mammal-ocr/src/rectangle_detection/page_0.jpg"

# Read the image using OpenCV
img = cv2.imread(image_path)

# Get the bounding box coordinates and text
config = r'--oem 3 --psm 6'
boxes_data = pytesseract.image_to_boxes(img, config=config, lang='eng')

# Draw bounding boxes on the image
for box in boxes_data.splitlines():
    box = box.split()
    if box[0] in " ":
        continue
    # Extract box coordinates and draw a rectangle
    x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    cv2.rectangle(img, (x, img.shape[0] - y), (w, img.shape[0] - h), (0, 255, 0), 1)
    # Draw the character or text found within the box
    cv2.putText(img, box[0], (x, img.shape[0] - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
import easyocr
import cv2

# Path to your image
image_path = "./page_0.jpg"

# Read the image using OpenCV
img = cv2.imread(image_path)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # You can specify languages here

# Perform OCR to get bounding boxes and text
result = reader.readtext(img, decoder='beamsearch')

# Draw bounding boxes on the image
for detection in result:
    top_left = tuple(map(int, detection[0][0]))
    bottom_right = tuple(map(int, detection[0][2]))
    text = detection[1]
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    img = cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
import boto3
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Initialize Textract client
textract = boto3.client('textract')

# Path to your image
image_path = "/Users/salarsatti/projects/mammal-ocr/src/rectangle_detection/page_1.jpg"

# Call Amazon Textract
response = textract.detect_document_text(
    Document={'Bytes': open(image_path, 'rb').read()}
)

# Load image
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

# Process Textract response
for item in response['Blocks']:
    if item['BlockType'] == 'WORD':
        # Draw bounding box around each word
        box = item['Geometry']['BoundingBox']
        width, height = image.size
        left = width * box['Left']
        top = height * box['Top']
        draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], outline='red')

        # Get the detected text
        detected_text = item['Text']

        # Display the text near the bounding box
        text_x = left
        text_y = top - 20  # Adjust this value based on text position preference
        draw.text((text_x, text_y), detected_text, fill='red')  # Text color is set to red

# Display image with bounding boxes around identified text and text labels
plt.imshow(image)
plt.axis('off')
plt.show()

# %%
import cv2
from transformers import TrOCRProcessor, TrOCRModel

# Load TrOCR model and preprocessor
model = TrOCRModel.from_pretrained("MODEL_NAME")
processor = TrOCRProcessor.from_pretrained("MODEL_NAME")

# Your image file path
image_path = "path/to/your/image.jpg"

# Read the image
image = cv2.imread(image_path)

# Preprocess the image and get OCR predictions
inputs = processor(image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)

# Get bounding boxes and text
predicted_boxes = outputs.prediction_boxes
predicted_text = outputs.prediction_text

# Draw bounding boxes on the image
for box in predicted_boxes[0]:
    start_point = (int(box[0]), int(box[1]))
    end_point = (int(box[2]), int(box[3]))
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

# Show the image with bounding boxes
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('image', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()