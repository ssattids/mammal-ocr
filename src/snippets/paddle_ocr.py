# %%
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import io
from PIL import Image, ImageDraw
import uuid
# %%
ocr = PaddleOCR(use_angle_cls=False, lang='en', rec=False) # need to run only once to download and load model into memory
with open(f"/Users/salarsatti/projects/mammal-ocr/src/snippets/page_0.jpg", "rb") as f:
    image_bytes = f.read()

result = ocr.ocr(image_bytes, cls=False)
# %%
image = Image.open(io.BytesIO(image_bytes))
width, height = image.size
# %%
draw = ImageDraw.Draw(image)
for i, box in enumerate(result[0]):
    box = np.array(box[0]).astype(np.int32)
    xmin = min(box[:, 0])
    ymin = min(box[:, 1])
    xmax = max(box[:, 0])
    ymax = max(box[:, 1])
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    # draw.text((xmin, ymin), f"{i}", fill="black")
# %%
display(image)
# %%

# %%
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

blocks
# %%
