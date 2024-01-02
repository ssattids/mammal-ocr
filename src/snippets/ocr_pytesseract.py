# %% 
import cv2
import pytesseract
# Read the image using OpenCV
image = cv2.imread('page_0.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

extracted_text = pytesseract.image_to_string(gray_image)

# Get bounding boxes around the text
boxes = pytesseract.image_to_boxes(gray_image)

# Draw bounding boxes on the original image
for box in boxes.splitlines():
    box = box.split()
    x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

# # Get bounding boxes around the text
# boxes = pytesseract.image_to_boxes(gray_image)

# # Draw bounding boxes on the original image
# for box in boxes.splitlines():
#     box = box.split()
#     x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
#     cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
# %%
# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
import pytesseract
from PIL import Image, ImageDraw

# Path to your Tesseract executable (if not in your PATH environment variable)

# Load the image using Pillow
image_path = './page_0.png'
image = Image.open(image_path)

# Perform OCR on the image to extract text and bounding boxes
data = pytesseract.image_to_boxes(image)

# Create a drawing object
draw = ImageDraw.Draw(image)

# Iterate through bounding box data and draw boxes on the image
for line in data.splitlines():
    character, x1, y1, x2, y2, *_ = line.split()
    # Convert coordinates to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Draw a rectangle around the character
    draw.rectangle([(x1, y1), (x2, y2)], outline="red")
#%%
# Save or display the image with bounding boxes

image.show() 
# %%
import pytesseract
from pytesseract import Output
import cv2
img = cv2.imread('page_0.png')

d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
# %%
import cv2
import pytesseract

filename = 'page_0.png'

# read the image and get the dimensions
img = cv2.imread(filename)
h, w, _ = img.shape # assumes color image

# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img) # also include any config options you use

# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

# show annotated image and wait for keypress
cv2.imshow(filename, img)
cv2.waitKey(0)
# %%
