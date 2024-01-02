# %%
from PIL import Image, ImageDraw
import json
import sklearn.neighbors as skneighbors
import json
import os
import matplotlib
import logging
import sys
from common import draw_bounding_box
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
import copy
# %%
def get_ann_label(anns):
    """
    Get the annotations with labels
    """
    ann_labels = []
    for i, ann in enumerate(anns):
        if ann['value'].get('labels') != None:
            ann_labels.append(ann)
    return ann_labels
# %%
def is_point_inside_polygon(point, polygon):
    # Create the Path object from the polygon vertices
    if type(point[0])==dict:
        point = (point['X'], point['Y'])
    if type(polygon[0])==dict:
        polygon = [(p['X'], p['Y']) for p in polygon]
    path = matplotlib.path.Path(polygon)
    # Check if the point=(x, y) is inside the polygon
    return path.contains_point(point)

def is_inside(rect1, rect2):
    # Check if all points of rect1 are inside rect2
    # rect1 and rect2 are lists of tuples containing four corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    if type(rect1[0])==dict:
        rect1 = [(p['X'], p['Y']) for p in rect1]
    if type(rect2[0])==dict:
        rect2 = [(p['X'], p['Y']) for p in rect2]
    for point in rect1:
        x, y = point
        if not (rect2[0][0] <= x <= rect2[2][0] and rect2[0][1] <= y <= rect2[2][1]):
            return False
    return True

def rectangle_intersect(bbox1, bbox2):
    """
        Calculate the intersection of two rectangle
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
    """
    if type(bbox[0])==dict:
        bbox = [(p['X'], p['Y']) for p in bbox]
    bbox_x1, bbox_y1 = bbox[0]
    bbox_x2, bbox_y2 = bbox[2]
    dx = abs(bbox_x2 - bbox_x1)
    dy = abs(bbox_y2 - bbox_y1)
    return dx*dy

def get_inside_rectangle(search_rectangle, inside_rectangle):
    """
        Search a typically smaller inside rectnagle inside a search rectangle

        If certain points lie outside of the search triangle, then default to
        the boundaried of the search rectanlge
        
    """
    sr, ir = search_rectangle, inside_rectangle
    srx1, irx1 = sr[0]['X'], ir[0]['X']
    sry1, iry1 = sr[0]['Y'], ir[0]['Y']
    srx2, irx2 = sr[2]['X'], ir[2]['X']
    sry2, iry2 = sr[2]['Y'], ir[2]['Y']
    if irx1 <= srx1:
        frx1 = srx1
    else:# irx1 > srx1:
        frx1 = irx1

    if iry1 <= sry1:
        fry1 = sry1
    else: # iry1 > sry1:
        fry1 = iry1

    if irx2 <= srx2:
        frx2 = irx2
    else: # irx2 > srx2:
        frx2 = srx2

    if iry2 <= sry2:
        fry2 = iry2
    else: # iry2 > sry2
        fry2 = sry2

    return [
        {"X":frx1, "Y":fry1},
        {"X":frx2, "Y":fry1},
        {"X":frx2, "Y":fry2},
        {"X":frx1, "Y":fry2},
    ]

def draw_rectangle(draw, rectangle_points, original_height=1, original_width=1, fill='red', width=9):

    RP = rectangle_points
    o_w = original_width
    o_h = original_height
    rectangle_line_points = [
        (RP[0]['X']*o_w, RP[0]['Y']*o_h),
        (RP[1]['X']*o_w, RP[1]['Y']*o_h),
        (RP[2]['X']*o_w, RP[2]['Y']*o_h),
        (RP[3]['X']*o_w, RP[3]['Y']*o_h),
        (RP[0]['X']*o_w, RP[0]['Y']*o_h),
    ]
    draw.line(rectangle_line_points, fill=fill, width=width)

# %%
def get_all_ann_labels(dir_path):
    if not os.path.exists(dir_path):
        raise Exception(f"Directory does not exist: {dir_path}")
    ann_labels = []
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            # Full path of the file
            file_path = os.path.join(root, file_name)
            if file_path.endswith(".json"):
                print(file_path)
                with open(file_path) as f:
                    data = json.load(f)
                    anns = data[0]["annotations"][0]["result"]
                    ann_labels += get_ann_label(anns)
    if ann_labels == []:
        raise Exception(f"Number of annotations: {len(ann_labels)}", )
    return ann_labels


# %%
# path
page_index = 3
dir_path = f"./data/masks/labelstudio_download/page_{page_index}"
ann_labels = get_all_ann_labels(dir_path)

# %%
# open the image
image_file_path = f"./data/masks/not_filled_images/page_{page_index}.jpg"
image = Image.open(image_file_path)

# %%
ocr_blocks_path = f"data/ocr/not_filled_blocks/page_{page_index}.json"
with open(ocr_blocks_path) as f:
    ocr_blocks = json.load(f)
no_fill_ocr_line_blocks = [block for block in ocr_blocks if block["BlockType"] == "LINE"] 

# %%
# iterate through the annotations which describe the area of the field
# keep track of the text in the fields that of a unfilled form
fields = []
draw = ImageDraw.Draw(image.copy())
for i, ann in enumerate(ann_labels):
    ann = ann.copy()
    # get the annotation geometric data to be processed
    val = ann['value'].copy()
    o_w = ann['original_width']
    o_h = ann['original_height']
    val['x'] = val['x']/100
    val['y'] = val['y']/100
    val['width'] = val['width']/100
    val['height'] = val['height']/100
    # get all the labels
    labels = ann['value']['labels']
    if len(labels) != 1:
        raise Exception("Only support one label per annotation")
    field_name = labels[0]
    logging.info(f"Processing field: {field_name}")
    # create field object - this should contain everything we need to extract data from the image
    field = {
        "name": field_name,
        "polygon": [],
        "contained_ocr_text_blocs": [],
    }
    # the allowable fillable text area for a field - convert to geometric polygon similar to textract OCR
    field_area_GP = [
        {'X':val['x'], 'Y':val['y']}, 
        {'X':val['x']+val['width'], 'Y':val['y']}, 
        {'X':val['x']+val['width'], 'Y':val['y']+val['height']}, 
        {'X':val['x'], 'Y':val['y']+val['height']}, 
    ]
    field["Geometry"] = {
        "Polygon" : field_area_GP,
    }
    # draw a bounding box around the field area
    draw_rectangle(draw, field_area_GP, original_width=o_w, original_height=o_h, fill='red', width=9)

    # iterate throught the non filled line blocks
    for line_block in no_fill_ocr_line_blocks:
        line_block_copy = copy.deepcopy(line_block)
        ocr_GP = line_block_copy["Geometry"]["Polygon"]
        field_GP = field['Geometry']['Polygon']
        # if the intersection of the OCR text is about 85% of the field area, add it as a contained block
        ocr_GP_area_intersection = rectangle_intersect(ocr_GP, field_GP)
        ocr_GP_area = rectangle_area(ocr_GP)
        # TODO update polygon to exist only in the field
        if ocr_GP_area_intersection/ocr_GP_area > 0.70:
            # draw a bounding box around the text that is contained in the field area
            draw_rectangle(draw, ocr_GP, original_width=o_w, original_height=o_h, fill='green', width=2)
            # keep track of everything found in the field area
            line_block_copy['Geometry']['Polygon'] = get_inside_rectangle(
                field_GP.copy(),
                ocr_GP
            )
            field["contained_ocr_text_blocs"].append(line_block_copy)
    fields.append(field)

# %%
# write the fields object
with open(f"./data/fields_data/page_{page_index}.json", "w") as f:
    json.dump(fields, f)
# %%
"""
##################################################
##################################################
Investigate outputs
##################################################
##################################################
"""
# %%
display(image)
# %%
draw_bounding_box(no_fill_ocr_line_blocks,
                  image,
                  block_type="LINE",
                  block_type_line_properties={"outline": "red", "fill": "blue"})
# %%
field_fax = [field for field in fields if field['name'] == 'agency_fax_number'][0].copy()
field_fax['BlockType'] = "LINE"

text_block_fax = [b for b in no_fill_ocr_line_blocks if "FAGANO" in b['Text'] ][0].copy()

rectangle_intersect(field_fax['Geometry']['Polygon'], text_block_fax['Geometry']['Polygon']) / rectangle_area(text_block_fax['Geometry']['Polygon'])

new_field = {
    'BlockType' : "LINE",
    "Geometry": {
        "Polygon": get_inside_rectangle(field_fax['Geometry']['Polygon'], text_block_fax['Geometry']['Polygon'])
    }
}
# %%
# ocr_fields_img = draw_bounding_box([field_fax, text_block_fax, new_field], image.copy())
ocr_fields_img = draw_bounding_box([field_fax, new_field], image.copy())

# %%
blocks = []
for f in fields:
    for cotb in f["contained_ocr_text_blocs"]:
        blocks.append(cotb)
ocr_fields_img = draw_bounding_box(blocks, image.copy())

# %%
ocr_blocks_img = draw_bounding_box(no_fill_ocr_line_blocks, image.copy())
# %%
