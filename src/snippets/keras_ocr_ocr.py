# %%
import keras_ocr
from PIL import Image, ImageDraw
# %%
# Load the detector and recognizer models from keras-ocr
pipeline = keras_ocr.pipeline.Pipeline()

# Replace 'path/to/your/image.jpg' with your image file path
image_path = 'page_0.jpg'
image = keras_ocr.tools.read(image_path)

# Perform text detection
prediction_groups = pipeline.recognize([image])
# %%
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
# %%
# Iterate through the detected text and draw bounding boxes
for predictions in prediction_groups:
    for _, box in predictions:
        # Extract box coordinates
        x_min, y_min = box[0]
        x_max, y_max = box[2]
        draw.rectangle([(int(x_min), int(y_min)), (int(x_max), int(y_max))], outline="red")
# %%


# %%
image.show()
# %%
