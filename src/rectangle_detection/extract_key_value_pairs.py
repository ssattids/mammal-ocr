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
import matplotlib
from common import draw_bounding_box, aws_textract_detect, convert_pil_to_bytes, get_all_page_data

def rectangle_intersect(bbox1, bbox2):
    """
        Calculate the intersection of two rectangle
        Input:
            bbox1: list of points of the first rectangle
            bbox2: list of points of the second rectangle
    """
    if type(bbox1[0])==dict:
        bbox1 = [(p['X'], p['Y']) for p in bbox1]
    if type(bbox2[0])==dict:
        bbox2 = [(p['X'], p['Y']) for p in bbox2]

    bbox1_x1, bbox1_y1 = bbox1[0]
    bbox1_x2, bbox1_y2 = bbox1[2]
    bbox2_x1, bbox2_y1 = bbox2[0]
    bbox2_x2, bbox2_y2 = bbox2[2]

    dx = min(bbox1_x2, bbox2_x2) - max(bbox1_x1, bbox2_x1)
    dy = min(bbox1_y2, bbox2_y2) - max(bbox1_y1, bbox2_y1)

    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0

def rectangle_area(bbox):
    """
        Calculate the area of a rectangle
        Input:
            bbox: list of points of the rectangle
    """
    if type(bbox[0])==dict:
        bbox = [(p['X'], p['Y']) for p in bbox]
    bbox_x1, bbox_y1 = bbox[0]
    bbox_x2, bbox_y2 = bbox[2]
    dx = abs(bbox_x2 - bbox_x1)
    dy = abs(bbox_y2 - bbox_y1)
    return dx*dy

def process_field_text(field, line_blocks):
    """
        For a field object, iterate over the ocr line blocks and returns the text that is inside the field area based on an intersection ratio
        Input:
            field: field object
            line_blocks: list of line blocks
    """
    text = ""
    for line_block in line_blocks:
        # check if it is inside the field area
        line_block_field_intersect = rectangle_intersect(
            field["Geometry"]["Polygon"],
            line_block["Geometry"]["Polygon"]
        )
        line_block_area = rectangle_area(line_block["Geometry"]["Polygon"])
        line_block_field_intersect_ratio = line_block_field_intersect/line_block_area
        # 65% of the new line area must intersect with the field area
        if line_block_field_intersect_ratio > 0.65:
            text += line_block["Text"] + "\n"
          
    return text
# %%
input_pdf = "/Users/salarsatti/projects/mammal-ocr/tests/test.pdf"
all_page_data = get_all_page_data(input_pdf, dpi=300, image_format='PNG')

field_kv = {}   
for page_index in range(len(all_page_data)):
    # if page_index != 0:
    #     continue
    image_bytes = all_page_data[page_index]["image_bytes"]
    pil_image = Image.open(io.BytesIO(image_bytes))
    image_width, image_height = pil_image.size

    # open json file
    non_filled_ocr_blocks_path = f"./data/ocr/not_filled_blocks/page_{page_index}.json"
    with open(non_filled_ocr_blocks_path) as json_file:
        non_filled_ocr_blocks = json.load(json_file)
    # use the non filled blocks to mask the image and leave only the values
    masked_image_pil = draw_bounding_box(
        non_filled_ocr_blocks,
        pil_image,
        block_type="LINE",
        block_type_line_properties={"outline":"black", "fill":"white"}
    )
    masked_image_bytes = convert_pil_to_bytes(masked_image_pil, format='PNG')
    # ocr the values
    r = aws_textract_detect(masked_image_bytes)
    masked_image_blocks = r["Blocks"]
    # get the line blocks
    masked_image_blocks_line = [b for b in masked_image_blocks if b['BlockType']=="LINE"]
    # get the fields blocks
    page_index_fields_path = f"./data/fields_data/page_{page_index}.json"
    with open(page_index_fields_path) as json_file:
        page_index_fields = json.load(json_file)
 
    for field in page_index_fields:
        # if field["name"] == "agency_name_and_address":
        text_value = process_field_text(field, masked_image_blocks_line)
        field_kv[field["name"]] = text_value

    # break

# %%
# VIEW THE KEY VALUE PAIRS EXTRACTED
for key, val in field_kv.items():
    if val != "":
        print("key:", key)
        print("val:", val)
        print("")