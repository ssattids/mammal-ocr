# %%
from common import get_all_page_data, convert_bytes_to_pil, crop_image, make_large_canvas, ocr_magic
from PIL import Image, ImageDraw
import fitz
import json

class OCRParse():
    def __init__(self):
        pass

    
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

def assert_found_blocks_size(blocks, expected_size, block_identifier):
    if (len(blocks) != expected_size):
        raise Exception(f"Found {len(blocks)} {block_identifier} blocks, expected {expected_size}")

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

def get_crop_start_stop_y_coords(
        pil_image,
        start_block,
        stop_block,
        start_inclusive=True,
        stop_inclusive=False):
    if start_inclusive:
        y_start = start_block['Geometry']['Polygon'][0]['Y']
    else:
        y_start = start_block['Geometry']['Polygon'][2]['Y']
    if stop_inclusive:
        y_stop = stop_block['Geometry']['Polygon'][2]['Y']
    else:
        y_stop = stop_block['Geometry']['Polygon'][0]['Y']

    x1, y1, x2, y2 = (
        0*pil_image.width,
        y_start*pil_image.height,
        1*pil_image.width,
        y_stop*pil_image.height,
    )
    print(x1, y1, x2, y2)
    cropped_coordinates = (x1, y1, x2, y2)
    return cropped_coordinates

class Accord130():
    def __init__(self):
        self.sections = []
        self.section_images = []

    def initialize(self):
        pass

    def indentify_sections(self, large_canvas):
        """
            Identify the various sections in the page
        """
        # PAGE_INDEX 0
        wce = find_blocks(large_canvas, "workers compensation application")
        assert_found_blocks_size(wce, 1, "workers compensation application")
        sos = find_blocks(large_canvas, "status of submission")
        assert_found_blocks_size(sos, 1, "status of submission")
        locs = find_blocks(large_canvas, "locations")
        assert_found_blocks_size(locs, 1, "locations")
        pi = find_blocks(large_canvas, "policy information")
        assert_found_blocks_size(pi, 1, "policy information")
        teap = find_blocks(large_canvas, "total estimated annual premium -")
        assert_found_blocks_size(teap, 1, "total estimated annual premium")
        ci = find_blocks(large_canvas, "contact information")
        assert_found_blocks_size(ci, 1, "contact information")
        iie = find_blocks(large_canvas, "individuals included")
        assert_found_blocks_size(iie, 1, "individuals included")
        p104 = find_blocks(large_canvas, "page 1 of 4")
        assert_found_blocks_size(p104, 1, "page 1 of 4")
        sections_data = [
            {
                "name": "agency_and_company_data",
                "start_block": wce[0],
                "end_block": sos[0],
                "page_index": "0",
                "offset_start": 0.0062
            },
            {
                "name": "status_of_submission",
                "start_block": sos[0],
                "end_block": locs[0],
                "page_index": "0",
                "offset_start": 0.0048
            },
            {
                "name": "locations",
                "start_block": locs[0],
                "end_block": pi[0],
                "page_index": "0",
                "offset_start": 0.0048
            },
            {
                "name": "policy_information",
                "start_block": pi[0],
                "end_block": teap[0],
                "page_index": "0",
                "offset_start": 0.0048
            },
            {
                "name": "total_annual_premium",
                "start_block": teap[0],
                "end_block": ci[0],
                "page_index": "0",
                "offset_start": 0.0048
            },
            {
                "name": "contact_information",
                "start_block": ci[0],
                "end_block": iie[0],
                "page_index": "0",
                "offset_start": 0.0048
            },
            {
                "name": "individuals_included",
                "start_block": iie[0],
                "end_block": p104[0],
                "page_index": "0",
                "offset_start": 0.0048
            },
        ]
        # PAGE_INDEX 1
        srs = find_blocks(large_canvas, "state rating sheet #")
        assert_found_blocks_size(srs, 1, "state rating sheet #")
        pris = find_blocks(large_canvas, "premium")
        p = find_blocks_range(pris, end_x_range=[0.0, 0.2], start_y_range=[.25, 0.5])
        assert_found_blocks_size(p, 1, "premium")
        ra1 = find_blocks(large_canvas, "remarks (acord 101")
        assert_found_blocks_size(ra1, 1, "remarks (acord 101)")
        p2o4 = find_blocks(large_canvas, "page 2 of 4")
        p2o4r = find_blocks_range(p2o4, start_y_range=[0.25, 0.5])
        assert_found_blocks_size(p2o4r, 1, "page 2 of 4")
        sections_data += [
            {
                "name": "state_rating_sheet",
                "start_block": srs[0],
                "end_block": p[0],
                "page_index": "1",
                "offset_start": 0.0048
            },
            {
                "name": "premium",
                "start_block": p[0],
                "end_block": ra1[0],
                "page_index": "1",
                "offset_start": 0.0048
            },
            {
                "name": "remarks_acord_101",
                "start_block": ra1[0],
                "end_block": p2o4r[0],
                "page_index": "1",
                "offset_start": 0.0048
            },
        ]       
        # PAGE_INDEX 2
        pci = find_blocks(large_canvas, "prior carrier information")
        assert_found_blocks_size(pci, 1, "prior carrier information")
        nob = find_blocks(large_canvas, "nature of business")
        assert_found_blocks_size(nob, 1, "nature of business")
        nob_0_y = nob[0]['Geometry']['Polygon'][0]['Y']
        gis = find_blocks(large_canvas, "general information") # general information blocks
        # search for general information title to be 
        # - in the first 20% of the width of the page
        # - search half a page down
        g1 = find_blocks_range(gis, start_x_range=[0,0.2], start_y_range=[nob_0_y, nob_0_y+0.125])
        assert_found_blocks_size(g1, 1, "general information")
        p3o4 = find_blocks(large_canvas, "page 3 of 4")
        assert_found_blocks_size(p3o4, 1, "page 3 of 4")

        sections_data += [
            {
                "name": "prior_carrier_information",
                "start_block": pci[0],
                "end_block": nob[0],
                "page_index": "2",
                "offset_start": 0.005
            },
            {
                "name": "nature_of_business",
                "start_block": nob[0],
                "end_block": g1[0],
                "page_index": "2",
                "offset_start": 0.005
            },
            {
                "name": "general_information",
                "start_block": g1[0],
                "end_block": p3o4[0],
                "page_index": "2",
                "offset_start": 0.005
            },
        ]
        # PAGE_INDEX 3
        gisc = find_blocks(large_canvas, "general information (continued")
        gisc = find_blocks_range(gisc, start_x_range=[0,0.1])
        assert_found_blocks_size(gisc, 1, "general information (continued")
        sign = find_blocks(large_canvas, "signature")
        sign = find_blocks_range(sign, end_x_range=[0,0.15])
        assert_found_blocks_size(sign, 1, "signature")
        p4o4 = find_blocks(large_canvas, "page 4 of 4")
        assert_found_blocks_size(p4o4, 1, "page 4 of 4")
        sections_data += [
            {
                "name": "general_information_continued",
                "start_block": gisc[0],
                "end_block": sign[0],
                "page_index": "3",
                "offset_start": 0.005
            },
            {
                "name": "signature",
                "start_block": sign[0],
                "end_block": p4o4[0],
                "page_index": "3",
                "offset_start": 0.005
            }
        ]

        return sections_data


    def detect(self, pdf_bytes):
        # open the pdf bytes

        # take each image and ocr it
        # make it into a really large canvas
        # identify the sections
        # get the key value pair 
        all_page_data = get_all_page_data(pdf_bytes, dpi=300)
        # OCR each page
        for page in all_page_data:
            page["blocks"] = ocr_magic(page["image_bytes"])
        
        large_canvas = make_large_canvas(all_page_data)
        large_canvas_image = make_large_canvas_image(all_page_data)
        # identify the sections
        self.sections = self.indentify_sections(large_canvas)
        # use the self.section to create a seperate image for each section
        
        # use the OCR information to blot out the keys in 

        # apply the masks to each section
        section_images = []
        for section in sections:
            section_image = crop_image(
                large_canvas_image, 
                (
                    0,
                    section['start_block']['Geometry']['Polygon'][0]['Y']*large_canvas_image.height,
                    large_canvas_image.width,
                    section['end_block']['Geometry']['Polygon'][0]['Y']*large_canvas_image.height,
                )
            )
            section_images.append(section_image)
        
        self.section_images = section_images

        
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
with open("/Users/salarsatti/projects/mammal-ocr/src/data/Acord_130_2_not_filled.pdf", "rb") as rb:
    pdf_bytes = rb.read()
# %%
all_page_data = get_all_page_data(pdf_bytes, dpi=300)
# %%
for page in all_page_data:
    page["blocks"] = ocr_magic(page["image_bytes"])
# %%
large_canvas = make_large_canvas(all_page_data)
large_canvas_image = make_large_canvas_image(all_page_data)
# %%
sections = Accord130().indentify_sections(large_canvas)
# %%
#############################################
def starts_with_one_of(string, array_of_strings):

    for prefix in array_of_strings:
        if string.startswith(prefix):
            return True
    return False
str_strs = [
    ["agency_", "company_"],
    ["submission_", "audit_", "billing_", "payment_"],
    ["locations_"],
    ["policy_information"],
    ["total_"],    
    ["contact_"],
    ["individuals_"],
    ["state_rating_", "state_ratings_"],
    ["premium_"],
    ["remarks"],
    ["prior_carrier_information_"],
    ["nature_of_business"],
    ["general_information"],
    ["general_information"],
    ["signature"]
]

for s_i, section in enumerate(sections):
    if s_i <= 12:
        start_block = section['start_block']
        end_block = section['end_block']
        cropped_coordinates = get_crop_start_stop_y_coords(large_canvas_image, start_block, end_block)
        section_image = crop_image(large_canvas_image, cropped_coordinates)
        display(section_image)
        section_fields = [field for field in all_fields if starts_with_one_of(field['name'], str_strs[s_i])]
        section_image_masks = section_image.copy()
        draw=ImageDraw.Draw(section_image_masks)
        # 
        mask_points = []
        for i, section_field in enumerate(section_fields):
            points = []
            for point in section_field['Geometry']['Polygon']:
                points.append((
                    large_canvas_image.width * point['X'], 
                    large_canvas_image.height * (point['Y']-start_block['Geometry']['Polygon'][2]['Y']+section['offset_start']-0.0025)
                ))
            mask_points.append(points)
            draw.polygon(
                (points),
                outline="red",
                fill="blue"
            )
        section_predict_info = {
            "name": section['name'],
            "section_image_coordinates": cropped_coordinates,
            "section_fields": section_fields,
            "mask_points": mask_points
        }
        display(section_image_masks)
        section_image = crop_image(
            large_canvas_image,
            section_predict_info["section_image_coordinates"]
        )
        display(section_image)
        # double check to make sure the coordinates make sense
        section_image_masks = section_image.copy()
        draw=ImageDraw.Draw(section_image_masks)
        for i, points in enumerate(section_predict_info["mask_points"]):
            draw.polygon(
                (points),
                outline="red",
                fill="blue"
            )
        # save data to the file
        with open(f'/Users/salarsatti/projects/mammal-ocr/src/rectangle_detection/Acord130/{s_i}.json', 'w') as outfile:
            json.dump(section_predict_info, outfile)
        display(section_image_masks)
        section_image.save(f'/Users/salarsatti/projects/mammal-ocr/src/rectangle_detection/Acord130/section_{s_i}.png')
        section_image_masks.save(f'/Users/salarsatti/projects/mammal-ocr/src/rectangle_detection/Acord130/section_mask_{s_i}.png')
        

    

# %%
section_predict_info = {
        "name": section['name'],
        "section_image_coordinates": cropped_coordinates,
        "section_fields": section_fields,
        "mask_points": mask_points
    }

# %%
# double checks
section_image = crop_image(large_canvas_image, section_predict_info["section_image_coordinates"])
section_image_masks = section_image.copy()
draw=ImageDraw.Draw(section_image_masks)
for i, points in enumerate(section_predict_info["mask_points"]):
    draw.polygon(
        (points),
        outline="red",
        fill="blue"
    )
display(section_image_masks)
# %%
display(section_image)
# %%
import cv2
import numpy as np

def detect_rectangles(path):
    # reading image
    img = cv2.imread(path)
    # converting image into grayscale image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # setting threshold of gray image 
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
    print(threshold)
    # using a findContours() function 
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approxes = [cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True) for contour in contours]
    approxes_4 = [approx for approx in approxes if len(approx)==4]

    polygons = []
    for approx in approxes_4:
        contour = approx
        if contour.shape == (4,1,2):
            xs_ys = contour.reshape(4,2)
            # print
            # print(xs_ys)
            polygon = [
                {'X':xs_ys[0][0], 'Y':xs_ys[0][1]},
                {'X':xs_ys[2][0], 'Y':xs_ys[0][1]},
                {'X':xs_ys[2][0], 'Y':xs_ys[2][1]},
                {'X':xs_ys[0][0], 'Y':xs_ys[2][1]},
            ]
            polygons.append(polygon)
            
    return polygons
    

polygons = detect_rectangles(f'/Users/salarsatti/projects/mammal-ocr/src/rectangle_detection/Acord130/section_12.png')

# %%
section_image_dr = section_image.copy()
draw=ImageDraw.Draw(section_image_dr)

section_image_area = section_image.height*section_image.width
for polygon in polygons:    
    polygon_area = (polygon[2]['X']-polygon[0]['X'])*(polygon[2]['Y']-polygon[0]['Y'])
    if polygon_area < section_image_area*0.85:
        points = []
        for point in polygon:
            points.append((point['X'], point['Y']))
        draw.polygon(
            (points),
            outline="red",
            fill="red"
        ) 
# %%
display(section_image_dr)


# %%
from common import convert_pil_to_bytes

import boto3
client = boto3.client('textract')
response = client.analyze_document(
    Document={
        'Bytes': convert_pil_to_bytes(section_image)
    },
    FeatureTypes=["TABLES"]
)

# %%
for item in response['Blocks']:
    if item['BlockType'] == 'TABLE':
        # Process table data as needed
        print(item['Text'])
# %%
