import torch
from PIL import Image
from transformers import OCRProcessor, OCRPipeline

# Load the model and processor
model_name = "microsoft/trocr-large-printed"
processor = OCRProcessor.from_pretrained(model_name)
model = OCRPipeline(model_name)

# Load your image
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)

# Perform OCR
outputs = model(image)

# Process the outputs to get bounding boxes and text
text_lines = outputs["text"]
bounding_boxes = outputs["coordinates"]

# Display identified text and bounding boxes
for text, bbox in zip(text_lines, bounding_boxes):
    print(f"Text: {text}")
    print(f"Bounding Box: {bbox}")  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    print("----")