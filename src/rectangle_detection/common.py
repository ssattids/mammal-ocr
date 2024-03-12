#%%
import boto3
import io
from PIL import Image, ImageDraw
import json
import paddleocr
import easyocr
import cv2
import numpy as np
import uuid
import fitz
import copy
# %%

def get_all_page_data(input_pdf, dpi=300, image_format='PNG'):
    """
        Get a list of page objects for a .pdf file
        Input:
            input_pdf: path to input pdf file
            dpi: dpi to use for image conversion
            image_format: image format to use for image conversion
        Output:
            all_page_data: list of page objects
    """
    if type(input_pdf) == str:
        doc = fitz.open(input_pdf)
    elif type(input_pdf) == bytes:
        doc = fitz.open(stream=input_pdf, filetype="pdf")
    else:
        raise Exception("input_pdf must be a path to a .pdf file or a bytes object")

    images_bytes = []
    images = []
    all_page_data = []
    image_format = 'JPG'
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
    return all_page_data


def aws_textract_detect(image_bytes:bytes):
    """
        Input:
            image_bytes: image_bytes to be passed to aws textract
        Output:
            response: response object from aws textract
    """
    # make call to aws textract
    client = boto3.client('textract')
    response = client.detect_document_text(
        Document={'Bytes': image_bytes}
    )
    return response

def aws_textract_analyze(image_bytes:bytes, api_types=['FORMS']):
    """
        image_bytes: image_bytes to be passed to aws textract
        api_types: some of the specific OCR extraction endpoints for aws textract e.g. ['FORMS']
    """
    client = boto3.client('textract')
    response = client.analyze_document(
        Document={
            'Bytes': image_bytes
        },
        FeatureTypes=api_types
    )
    return response

def paddle_ocr_detect(image_bytes):
    """
        image_bytes: image_bytes to be passed to paddle ocr
    """
    # technically only once to download and load model into memory
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size

    ocr = paddleocr.PaddleOCR(use_angle_cls=False, lang='en', rec=False) # need to run 
    result = ocr.ocr(image_bytes, cls=False)

    blocks = []
    for i, box in enumerate(result[0]):
        box_coords = np.array(box[0]).astype(np.int32)
        xmin = min(box_coords[:, 0]) / width
        ymin = min(box_coords[:, 1]) / height
        xmax = max(box_coords[:, 0]) / width
        ymax = max(box_coords[:, 1]) / height
        textract_type_block = {
            "BlockType": "LINE",
            "Confidence": box[1][1],
            "Text": box[1][0],
            "Geometry": {
                "Polygon": [
                    {"X":xmin, "Y":ymin},
                    {"X":xmax, "Y":ymin},
                    {"X":xmax, "Y":ymax},
                    {"X":xmin, "Y":ymax},
                ]
            },
            "Id": str(uuid.uuid4())
        }

        blocks.append(textract_type_block)

    return {"Blocks": blocks}

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

def convert_pil_to_bytes(image_pil:Image, format:str="PNG"):
    """
        Function to convert a PIL Image object to bytes
        Input:
            image_pil: PIL Image object
            format: format to convert to, e.g. PNG, JPG
        Output:
            image_bytes: bytes object of the image
    """
    byte_io = io.BytesIO()
    image_pil.save(byte_io, format=format)
    image_bytes = byte_io.getvalue()
    return image_bytes

def convert_bytes_to_pil(image_bytes:bytes):
    """
        Function to convert bytes to a PIL Image object
        Input:
            image_bytes: bytes object of the image
        Output:
            image_pil: PIL Image object
    """
    image_pil = Image.open(io.BytesIO(image_bytes))
    return image_pil

def draw_bounding_box(blocks, image:Image, block_type=None, block_type_line_properties={}, block_type_word_properties={}):
    """
        blocks: blocks returned from aws textract
        image: PIL image to draw bounding boxes on
        block_type: type of block to draw bounding boxes on, e.g. LINE, WORD
        block_type_line_properties: properties to draw bounding boxes on, e.g. {"outline": "red", "fill": "blue"}
        block_type_word_properties: properties to draw bounding boxes on, e.g. {"outline": "red", "fill": "blue"}
    """
    image = image.copy()
    width = image.width
    height = image.height

    colors = ['red', 'blue', 'black', 'green']
    line_color = 0
    for i, block in enumerate(blocks):
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
        if block_type == "WORD" or block_type == None:
            if block['BlockType'] == "WORD":
                points=[]

                for polygon in block['Geometry']['Polygon']:
                    points.append((width * polygon['X'], height * polygon['Y']))

                if block_type_word_properties == {}:
                    draw.polygon((points), outline=colors[line_color%4])
                    line_color+=1
                else:
                    draw.polygon(
                        (points),
                        outline=block_type_word_properties.get("outline"),
                        fill=block_type_word_properties.get("fill")
                    ) 

                
        # Draw box around entire LINE  
        if block_type == "LINE" or block_type == None:
            if block['BlockType'] == "LINE":
                points=[]

                for polygon in block['Geometry']['Polygon']:
                    points.append((width * polygon['X'], height * polygon['Y']))

                if block_type_line_properties == {}:
                    draw.polygon((points), outline=colors[line_color%4])
                    line_color+=1    
                else:
                    draw.polygon(
                        (points),
                        outline=block_type_line_properties.get("outline"),
                        fill=block_type_line_properties.get("fill")
                    )

    # Display the image
    return image

def crop_image(img, coordinates):
    """
    Crop image based on coordinates and save the cropped portion to a new image file.
    
    Arguments:
    image_path: The path to the input image.
    coordinates: A tuple containing the coordinates (left, upper, right, lower) to define the cropping region.
    output_path: The path to save the cropped image.
    """
    # Open the image
    
    # Crop the image based on coordinates
    cropped_img = img.crop(coordinates)
    
    return cropped_img


def ocr_magic(image_bytes, block_type="LINE"):
    """
        Function to perform OCR on an image and return the blocks of a specific type

        Currently we OCR the image twice to get all the blocks 

        Input:
            image_bytes: the png bytes of an image
        Output:
            blocks: the blocks of the specified type
    """
    r = aws_textract_detect(image_bytes)
    # r = paddle_ocr_detect(image_bytes)
    blocks = r['Blocks']
    # white out the already identified blocks
    image = Image.open(io.BytesIO(image_bytes))
    bb_image = draw_bounding_box(
        blocks,
        image,
        block_type="LINE",
        block_type_line_properties={"outline":None, "fill":"white"}
    )
    # send the image again to be OCRed
    bb_image_bytes = convert_pil_to_bytes(bb_image, format="PNG")
    r_bb = aws_textract_detect(bb_image_bytes)
    blocks_bb = r_bb["Blocks"]
    # combine to line blocks
    if block_type != None:
        line_blocks = [b for b in blocks if b["BlockType"] == block_type]
        line_bb_blocks = [b for b in blocks_bb if b["BlockType"] == block_type]
    else:
        all_line_blocks = blocks + blocks_bb
    
    all_line_blocks = line_blocks + line_bb_blocks
    # white out all the blocks and show what we have
    bb_image = draw_bounding_box(
        all_line_blocks,
        image,
        block_type="LINE",
        block_type_line_properties={"outline":None, "fill":"white"}
    )
    return all_line_blocks

def update_block_large_canvas(block, page_index, num_pages):
    """
        Helper function called by get_relevant blocks
        it will update an identified text block position in such a way that the all pages in a document are treated as a large canvas
    """
    a_block_copy = copy.deepcopy(block)
    a_block_copy["Geometry"]["Polygon"][0]["Y"] = (a_block_copy["Geometry"]["Polygon"][0]["Y"] + page_index) / num_pages
    a_block_copy["Geometry"]["Polygon"][1]["Y"] = (a_block_copy["Geometry"]["Polygon"][1]["Y"] + page_index) / num_pages
    a_block_copy["Geometry"]["Polygon"][2]["Y"] = (a_block_copy["Geometry"]["Polygon"][2]["Y"] + page_index) / num_pages
    a_block_copy["Geometry"]["Polygon"][3]["Y"] = (a_block_copy["Geometry"]["Polygon"][3]["Y"] + page_index) / num_pages

    return a_block_copy

def make_large_canvas(pages):
    """
        Function that takes in an array of page objects, and returns an array of blocks that contain all the pages blocks as if it were on a large canvas
    """
    all_updated_blocks = []
    num_pages = len(pages)
    for page in pages:
        page_index = page["page_index"]
        page_blocks = page["blocks"]
        updated_blocks = []
        for block in page_blocks:
            updated_blocks.append(update_block_large_canvas(block, page_index, num_pages))
        all_updated_blocks += updated_blocks
    return all_updated_blocks

def make_large_canvas_page(pages):
    all_updated_blocks = []
    for page in pages:
        page_index = page["page_index"]
        page_blocks = page["blocks"]
        for block in page_blocks:
            updated_block = block.copy()
            updated_block["page_index"] = page_index
            all_updated_blocks.append(updated_block)
    return all_updated_blocks

# %%
