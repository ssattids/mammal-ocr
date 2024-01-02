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
dpi = 600
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
    # image.save(f"./page_{page_index}.png")
    break
    # display(image)
# %%

from PIL import Image

def crop_image(image, crop_box):
    """
        image: PIL Image object
        crop_box: (x_min, y_min, x_max, y_max)
    """
    cropped_image = image.copy().crop(crop_box)
    display(cropped_image)
    return cropped_image




# %%
import easyocr
import cv2
import numpy as np
import io
import uuid
# # Path to your image
# image_path = "./page_0.jpg"
# # Read the image using OpenCV
# img = cv2.imread(image_path)
def easy_ocr_detect(image_bytes):
    image_np = np.asarray(
        bytearray(io.BytesIO(image_bytes).read()), 
        dtype=np.uint8
    )
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img, decoder='beamsearch')
    blocks = []
    for detection in result:
        coords = detection[0]
        textract_type_block = {
            "BlockType": "LINE",
            "Confidence": detection[2],
            "Text": detection[1],
            "Geometry": {
                "Polygon": [
                    {
                        "X":min(coords[0][0], coords[3][0]),
                        "Y":min(coords[0][1], coords[1][1])
                    },
                    {
                        "X":max(coords[1][0], coords[2][0]),
                        "Y":min(coords[0][1], coords[1][1])
                    },
                    {
                        "X":max(coords[1][0], coords[2][0]),
                        "Y":max(coords[2][1], coords[3][1])
                    },
                    {
                        "X":min(coords[0][0], coords[3][0]),
                        "Y":max(coords[2][1], coords[3][1])
                    },
                ]
            },
            "Id": str(uuid.uuid4())
        }
        blocks.append(textract_type_block)
    return {"Blocks": blocks}

image_np = np.asarray(bytearray(io.BytesIO(image_bytes).read()), dtype=np.uint8)
img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

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
# cv2.imshow('Image with Bounding Boxes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# %%
pil_image = Image.fromarray(img.astype('uint8'))
display(pil_image)

# %%
box = (5781, 288, 6455, 378)
crop_image(pil_image, box)
# %%
cv2.imwrite(f'easy_ocr_page_0_dpi_{dpi}.jpg', img)
# %%