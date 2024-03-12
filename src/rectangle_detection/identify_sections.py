# %%
import os
import json
from PIL import Image, ImageDraw
from common import draw_bounding_box, aws_textract_detect, convert_pil_to_bytes, convert_bytes_to_pil, get_all_page_data, ocr_magic, make_large_canvas, crop_image

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
def get_all_ann_labels(dir_path):
    """
    Get all the annotations with labels from a directory of json files
    """
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

def get_ocr_type_block(ann, page_index, len_pages):
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
    # create field object - this should contain everything we need to extract data from the image
    field = {
        "name": field_name,
        "polygon": [],
        "contained_ocr_text_blocs": [],
    }
    field_area_GP = [
        {'X':val['x'], 'Y':(val['y'] + page_index) / len_pages }, 
        {'X':val['x']+val['width'], 'Y':(val['y'] + page_index) / len_pages}, 
        {'X':val['x']+val['width'], 'Y':(val['y']+val['height'] + page_index) / len_pages}, 
        {'X':val['x'], 'Y':(val['y']+val['height'] + page_index) / len_pages}, 
    ]
    field["Geometry"] = {
        "Polygon" : field_area_GP,
    }
    return field

all_fields = []
len_pages = len(range(4))
for page_index in range(len_pages):
    dir_path = f"./data/masks/labelstudio_download/page_{page_index}"
    ann_labels = get_all_ann_labels(dir_path)
    for ann in ann_labels:
        field = get_ocr_type_block(ann, page_index, len_pages)
        all_fields.append(field)

# %%
input_pdf = "/Users/salarsatti/projects/mammal-ocr/tests/end_to_end_test/test0.pdf"
all_page_data = get_all_page_data(input_pdf, dpi=300, image_format='PNG')
# %%
# OCR each page
for page in all_page_data:
    page["blocks"] = ocr_magic(page["image_bytes"])
# %%
def make_large_canvas_image(all_page_data):
    images = [convert_bytes_to_pil(page["image_bytes"]) for page in all_page_data]
    # Calculate the dimensions of the new image
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    # Create a new blank image
    new_image = Image.new('RGB', (max_width, total_height))
    # Paste images one below the other
    y_offset = 0
    for img in images:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height
    # Save the combined image
    return new_image
# %%
# make large canvas
large_canvas = make_large_canvas(all_page_data)
large_canvas_image = make_large_canvas_image(all_page_data)
# %%
def find_blocks(large_canvas, sub_text):
    blocks = []
    for b in large_canvas:
        if sub_text in b['Text'].lower():
            blocks.append(b)
    return blocks

def find_blocks_range(large_canvas, start_x_range=None, start_y_range=None, end_x_range=None, end_y_range=None):
    ranged_blocks = []
    for b in large_canvas:
        polygon_point_start = b['Geometry']['Polygon'][0]
        if start_x_range != None:
            # if a block is not in the x range, skip it
            if polygon_point_start['X'] < start_x_range[0] or polygon_point_start['X'] > start_x_range[1]:
                continue
        if start_y_range != None:
            # if a block is not in the y range, skip it
            if polygon_point_start['Y'] < start_y_range[0] or polygon_point_start['Y'] > start_y_range[1]:
                continue
        polygon_point_end = b['Geometry']['Polygon'][2]
        if end_x_range != None:
            # if a block is not in the x range, skip it
            if polygon_point_end['X'] < end_x_range[0] or polygon_point_end['X'] > end_x_range[1]:
                continue
        if end_y_range != None:
            # if a block is not in the y range, skip it
            if polygon_point_end['Y'] < end_y_range[0] or polygon_point_end['Y'] > end_y_range[1]:
                continue
        
        ranged_blocks.append(b)

    return ranged_blocks
# %%
def crop_start_stop_y(
        pil_image,
        start_block,
        stop_block,
        start_inclusive=False,
        stop_inclusive=False):
    if start_inclusive:
        y_start = start_block[0]['Geometry']['Polygon'][0]['Y']
    else:
        y_start = start_block[0]['Geometry']['Polygon'][2]['Y']
    if stop_inclusive:
        y_stop = stop_block[0]['Geometry']['Polygon'][2]['Y']
    else:
        y_stop = stop_block[0]['Geometry']['Polygon'][0]['Y']

    x1, y1, x2, y2 = (
        0*pil_image.width,
        y_start*pil_image.height,
        1*pil_image.width,
        y_stop*pil_image.height,
    )
    print(x1, y1, x2, y2)
    cropped = crop_image(pil_image, (x1, y1, x2, y2))
    return cropped

# %%
img = convert_bytes_to_pil(all_page_data[0]['image_bytes'])
wce = find_blocks(large_canvas, "workers compensation application")
assert(len(wce) == 1)
sos = find_blocks(large_canvas, "status of submission")
assert(len(sos) == 1)
locs = find_blocks(large_canvas, "locations")
assert(len(locs) == 1)
pi = find_blocks(large_canvas, "policy information")
assert(len(pi) == 1)
teap = find_blocks(large_canvas, "total estimated annual premium -")
assert(len(teap) == 1)
ci = find_blocks(large_canvas, "contact information")
assert(len(ci) == 1)
iie = find_blocks(large_canvas, "individuals included")
assert(len(iie) == 1)
poof = find_blocks(large_canvas, "page 1 of 4")
assert(len(poof) == 1)
# %%
def starts_with_one_of(string, array_of_strings):
    for prefix in array_of_strings:
        if string.startswith(prefix):
            return True
    return False

def get_section_masks(
        img, all_fields, start_block, end_block, 
        start_inclusive, end_inclusive,
        start_with_strs,
        offset
        ):
    section_img = crop_start_stop_y(img, start_block, end_block, start_inclusive=start_inclusive, stop_inclusive=end_inclusive)
    section_fields = [field for field in all_fields if starts_with_one_of(field['name'], start_with_strs)]
    draw=ImageDraw.Draw(section_img)

    for i, section_field in enumerate(section_fields):
        points = []
        for point in section_field['Geometry']['Polygon']:
            points.append((
                img.width * point['X'], 
                img.height * (point['Y']-start_block[0]['Geometry']['Polygon'][2]['Y']+offset)
            ))
        draw.polygon(
            (points),
            outline="red",
            fill="blue"
        )

    display(section_img)
# %%
# PAGE 1
get_section_masks(large_canvas_image, all_fields, wce, sos, True, False, ["agency_", "company_"], 0.0062)
# %%
get_section_masks(large_canvas_image, all_fields, sos, locs, True, False, ["submission_", "audit_", "billing_", "payment_"], 0.0048)
# %%
get_section_masks(large_canvas_image, all_fields, locs, pi, True, False, ["locations_"], 0.0048)
# %%
get_section_masks(large_canvas_image, all_fields, pi, teap, True, False, ["policy_"], 0.0048)
# %%
get_section_masks(large_canvas_image, all_fields, teap, ci, True, False, ["total_"], 0.0048)
# %%
get_section_masks(large_canvas_image, all_fields, ci, iie, True, False, ["contact_"], 0.0048)
# %%
get_section_masks(large_canvas_image, all_fields, iie, poof, True, False, ["individuals_"], 0.0048)
# %%
# %%
# PAGE 2
srs = find_blocks(large_canvas, "state rating sheet #")
assert(len(srs) == 1)
# %%
pris = find_blocks(large_canvas, "premium")
p = find_blocks_range(pris, end_x_range=[0.0, 0.2], start_y_range=[.25, 0.5])
assert(len(p) == 1)
# %%
ra1 = find_blocks(large_canvas, "remarks (acord 101")
assert(len(ra1) == 1)
# %%
p2o4 = find_blocks(large_canvas, "page 2 of 4")
p2o4r = find_blocks_range(p2o4, start_y_range=[0.25, 0.5])
assert(len(p2o4r) == 1)
# %%
get_section_masks(large_canvas_image, all_fields, srs, p, True, False, ["state_rating_", "state_ratings_"], 0.0048)
# %%
get_section_masks(large_canvas_image, all_fields, p, ra1, True, False, ["premium_"], 0.0048)
# %%
get_section_masks(large_canvas_image, all_fields, ra1, p2o4r, True, False, ["remarks"], 0.0054)
# %%
# PAGE 3
pci = find_blocks(large_canvas, "prior carrier information")
assert(len(pci) == 1)
nob = find_blocks(large_canvas, "nature of business")
assert(len(nob) == 1)
nob_0_y = nob[0]['Geometry']['Polygon'][0]['Y']
gis = find_blocks(large_canvas, "general information") # general information blocks
# search for general information title to be 
# - in the first 20% of the width of the page
# - search half a page down
g1 = find_blocks_range(gis, start_x_range=[0,0.2], start_y_range=[nob_0_y, nob_0_y+0.125])
assert(len(g1) == 1)
p3o4 = find_blocks(large_canvas, "page 3 of 4")
assert(len(p3o4) == 1)
# %%
get_section_masks(large_canvas_image, all_fields, pci, nob, True, False, ["prior_carrier_information_"], 0.005)
# %%
get_section_masks(large_canvas_image, all_fields, nob, g1, True, False, ["nature_of_business"], 0.005)
# %%
get_section_masks(large_canvas_image, all_fields, g1, p3o4, True, False, ["general_information"], 0.005)
# %%
# PAGE 4
gisc = find_blocks(large_canvas, "general information (continued")
gisc = find_blocks_range(gisc, start_x_range=[0,0.1])
assert(len(gisc) == 1)
sign = find_blocks(large_canvas, "signature")
sign = find_blocks_range(sign, end_x_range=[0,0.15])
assert(len(sign) == 1)
p4o4 = find_blocks(large_canvas, "page 4 of 4")
assert(len(p4o4) == 1)
# %%
get_section_masks(large_canvas_image, all_fields, gisc, sign, True, False, ["general_information"], 0.005)
# %%
get_section_masks(large_canvas_image, all_fields, sign, p4o4, True, False, ["signature"], 0.005)
# %%
