"""
    Identify valid form keys aread
    Take an empty form - and use the bounding boxes to identify the areas for they keys - the rest of the areas will be for the values
"""
# %%
import boto3
import io
from PIL import Image, ImageDraw
import json
import fitz
from common import aws_textract_detect, get_all_page_data, draw_bounding_box, convert_pil_to_bytes
# %%
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
    line_blocks = [b for b in blocks if b["BlockType"] == block_type]
    line_bb_blocks = [b for b in blocks_bb if b["BlockType"] == block_type]
    all_line_blocks = line_blocks + line_bb_blocks
    # white out all the blocks and show what we have
    bb_image = draw_bounding_box(
        all_line_blocks,
        image,
        block_type="LINE",
        block_type_line_properties={"outline":None, "fill":"white"}
    )
    return all_line_blocks
# %%
input_pdf = "/Users/salarsatti/projects/mammal-ocr/src/data/Travelers-RMD-MO-130-133-Forms_removed.pdf"
# get the page data for the pdf
all_page_data = get_all_page_data(input_pdf, dpi=300, image_format='PNG')
pil_images = []
for page_index in range(len(all_page_data)):
    page_data = all_page_data[page_index]
    image = Image.open(io.BytesIO(page_data["image_bytes"]))
    image.save(f"./data/ocr/not_filled_images/page_{page_index}.png")
    pil_images.append(image)
# %%
for page_index in range(len(all_page_data)):
    # page_index=0

    image_bytes=all_page_data[page_index]["image_bytes"]
    
    blocks = ocr_magic(image_bytes, block_type="LINE")
    
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_bb_bb = draw_bounding_box(
        blocks,
        image_pil,
        block_type="LINE",
        block_type_line_properties={"outline":"black", "fill":"white"}
    )
    image_bb_bb.save(f"./data/ocr/not_filled_images_bb/page_{str(page_index)}.png")

    with open(f"./data/ocr/not_filled_blocks/page_{str(page_index)}.json", 'w') as json_file:
        json.dump(blocks, json_file, indent=4)

# %%
