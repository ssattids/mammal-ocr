
# %%
import boto3
import io
import fitz
from PIL import Image, ImageDraw
import json
import numpy as np
import pandas as pd
import json
import random
from common import get_all_page_data
# %%
input_pdf = "/Users/salarsatti/projects/mammal-ocr/src/data/Acord_130_2_not_filled.pdf"
# input_pdf = "/Users/salarsatti/projects/mammal-ocr/src/data/ACORD_130.pdf"
# %%
# open pdf and loop over images
all_page_data = get_all_page_data(input_pdf, dpi=400, image_format='PNG')
pil_images = []
for page_index in range(len(all_page_data)):
    page_data = all_page_data[page_index]
    image = Image.open(io.BytesIO(page_data["image_bytes"]))
    image.save(f"./data/masks/not_filled_images/page_{page_index}.PNG")
    pil_images.append(image)
    display(image)

# %%    
# %%
# for page_index in range(len(images)):
page_index = 2
image = pil_images[page_index].copy()
width, height = image.size 

# %%
import cv2 
import numpy as np 
# reading image 
img = cv2.imread(f'./data/masks/not_filled_images/page_{page_index}.jpg') 
# converting image into grayscale image 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# setting threshold of gray image 
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
# using a findContours() function 
contours, _ = cv2.findContours( 
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

# %%
# get contours with approximate 4 shapes
approxes = [cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) for contour in contours]
approxes_4 = [approx for approx in approxes if len(approx)==4]
for approx in approxes_4:
    contour = approx
    if contour.shape[0]==4:
        img = cv2.polylines(img, [contour], isClosed=True, color=(0,0,255), thickness=10)
# %%
# displaying the image after drawing contours 
cv2.imshow('shapes', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
# %%
def convert_contour_to_labelstudio_json(original_width, original_height, contour, id:str):
    """
        Convert a vc2 contour to label studio json OCR label object
    """
    if contour.shape != (4,1,2):
        raise Exception("Countour shape should be (4,1,2)")
    xs_ys = contour.reshape(4,2)
    label_studio_json = [
            {
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation" : 0,
                "value" : {
                    "x": round(xs_ys[0][0]/original_width*100, 4),
                    "y": round(xs_ys[0][1]/original_height*100, 4),
                    "width": round((xs_ys[2][0] - xs_ys[0][0])/original_width*100, 4),
                    "height": round((xs_ys[2][1] - xs_ys[0][1])/original_height*100, 4),
                    "rotation": 0
                },
                "id":id,
                "from_name":"bbox",
                "to_name": "image",
                "type": "rectangle"
            },
            {
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation" : 0,
                "value" : {
                    "x": round(xs_ys[0][0]/original_width*100, 4),
                    "y": round(xs_ys[0][1]/original_height*100, 4),
                    "width": round((xs_ys[2][0] - xs_ys[0][0])/original_width*100, 4),
                    "height": round((xs_ys[2][1] - xs_ys[0][1])/original_height*100, 4),
                    "rotation": 0,
                    "labels": [
                      "Text"
                   ]
                },
                "id":id,
                "from_name":"label",
                "to_name": "image",
                "type": "labels"
            },
            {
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation" : 0,
                "value" : {
                    "x": round(xs_ys[0][0]/original_width*100, 4),
                    "y": round(xs_ys[0][1]/original_height*100, 4),
                    "width": round((xs_ys[2][0] - xs_ys[0][0])/original_width*100, 4),
                    "height": round((xs_ys[2][1] - xs_ys[0][1])/original_height*100, 4),
                    "rotation": 0,
                    "text": [
                      ""
                   ]
                },
                "id": id,
                "from_name": "transcription",
                "to_name": "image",
                "type": "textarea"
            }
    ]
    return label_studio_json

def generate_len_n_id(n=10):
    """
        Generate unique ID of length 10
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return ''.join(random.choices(alphabet, k=n))

convert_contour_to_labelstudio_json(width, height, contour, generate_len_n_id(n=10))

# %%
all_preds = []
for i, contour in enumerate(approxes_4):
    pred = convert_contour_to_labelstudio_json(width, height, contour, generate_len_n_id(n=10))
    all_preds+=pred
    
final_json = {
    "data": {
        "ocr": "path_to_be_given"
    },
    "annotations": [
        {
            "result": all_preds
        }
    ]
}
# %%
with open(f"./data/masks/labelstudio_upload/page_{page_index}.json", 'w') as json_file:
    json.dump(final_json, json_file)
# %%
