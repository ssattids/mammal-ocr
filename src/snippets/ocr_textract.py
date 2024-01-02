# %%
import boto3
from PIL import Image, ImageDraw


def aws_textract(image_bytes):
    # make call to aws textract
    client = boto3.client('textract')
    response = client.detect_document_text(
        Document={'Bytes': image_bytes}
    )
    return response

def aws_textract(image_bytes, api_types=['FORMS']):
    """
        image_bytes: image_bytes
        api_types: e.g. ['FORMS']
    """
    client = boto3.client('textract')
    response = client.analyze_document(
        Document={
            'Bytes': image_bytes
        },
        FeatureTypes=api_types
    )
    return response

def draw_bounding_box(blocks, image):
    image = image.copy()
    width = image.width
    height = image.height
    for block in blocks:
        # Display information about a block returned by text detection
        # print('Type: ' + block['BlockType'])
        # if block['BlockType'] != 'PAGE':
            # print('Detected: ' + block['Text'])
            # print('Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

        # print('Id: {}'.format(block['Id']))
        # if 'Relationships' in block:
        #     print('Relationships: {}'.format(block['Relationships']))
        # print('Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
        # print('Polygon: {}'.format(block['Geometry']['Polygon']))
        # print()
        draw=ImageDraw.Draw(image)
        # Draw WORD - Green -  start of word, red - end of word
        # if block['BlockType'] == "WORD":
        #     draw.line([(width * block['Geometry']['Polygon'][0]['X'],
        #     height * block['Geometry']['Polygon'][0]['Y']),
        #     (width * block['Geometry']['Polygon'][3]['X'],
        #     height * block['Geometry']['Polygon'][3]['Y'])],fill='green',
        #     width=2)
        
        #     draw.line([(width * block['Geometry']['Polygon'][1]['X'],
        #     height * block['Geometry']['Polygon'][1]['Y']),
        #     (width * block['Geometry']['Polygon'][2]['X'],
        #     height * block['Geometry']['Polygon'][2]['Y'])],
        #     fill='red',
        #     width=2)    

                
        # Draw box around entire LINE  
        if block['BlockType'] == "LINE":
            points=[]

            for polygon in block['Geometry']['Polygon']:
                points.append((width * polygon['X'], height * polygon['Y']))

            draw.polygon((points), outline="black", fill=None)    

    # Display the image
    display(image)
    return image

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
dpi = 300
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
r = aws_textract(image_bytes)
blocks = r["Blocks"]
# %%
image_bb = draw_bounding_box(blocks, image)
# %%
byte_io = io.BytesIO()
image_bb.save(byte_io, format='PNG')  
image_bb_bytes = byte_io.getvalue()
# %%
r_bb = aws_textract(image_bb_bytes)
blocks_bb = r_bb["Blocks"]
# %%
image_bb_bb = draw_bounding_box(blocks_bb, image_bb)

# %%
