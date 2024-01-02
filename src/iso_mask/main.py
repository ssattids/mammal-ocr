
# %%
import boto3
import io
import fitz
from PIL import Image, ImageDraw
import json
import sklearn.neighbors as skneighbors
import numpy as np
import pandas as pd
# %%
input_pdf = "/Users/salarsatti/projects/mammal-ocr/src/data/Travelers-RMD-MO-130-133-Forms.pdf"
# input_pdf = "/Users/salarsatti/projects/mammal-ocr/src/data/ACORD_130.pdf"
dpi = 400
zoom = 3 # to increase the resolution
mat = fitz.Matrix(dpi/72, dpi/72)
# %%
# open pdf and loop over images
doc = fitz.open(input_pdf)
images_bytes = []
images = []
all_page_data = []
image_format = 'PNG'
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
    # image.save(f"page_{page_index}.png")
    # display(image)
    break
# %%
page = 0
image = images[page].copy()
width, height = image.size 
image.save("output.png")
# %%

def draw_bounding_box_table(image, img_tables, width, height):
    
    image = image.copy()

    draw=ImageDraw.Draw(image)

    for img_table in img_tables:
        points = ([
            (img_table.bbox.x1, img_table.bbox.y1),
            (img_table.bbox.x2, img_table.bbox.y1),
            (img_table.bbox.x2, img_table.bbox.y2),
            (img_table.bbox.x1, img_table.bbox.y2),
            (img_table.bbox.x1, img_table.bbox.y1),

        ])
        draw.line(points, fill="red", width=9)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill="red")

    display(image)

def draw_masks(image, field_points, width, height):

    image = image.copy()
    draw=ImageDraw.Draw(image)

    for field_name, points in field_points.items():
        draw.line(points, fill="red", width=9)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill="red")

    display(image)

def absolute_to_relative_x1(x1_abs, width):
    return width/x1_abs
def absolute_to_relative_y1(y1_abs, height):
    return height/y1_abs
def absolute_to_relative_x2_plus_pos(x2_plus_pos, width):
    return width/x2_plus_pos
def absolute_to_relative_x2_plus_pos(y2_plus_pos, height):
    return height/y2_plus_pos




x1_abs = 100
y1_abs = 270
x2_plus_pos = 1250
y2_plus_pos = 390

x1_plot, y1_plot = (
    width/absolute_to_relative_x1(x1_abs, width),
    height/absolute_to_relative_y1(y1_abs, height)
)
x2_plot, y2_plot = (
    x1_plot+width/absolute_to_relative_x2_plus_pos(x2_plus_pos, width),
    y1_plot+height/absolute_to_relative_x2_plus_pos(y2_plus_pos, height)
)
agency_name_and_address_mask = ([
    (x1_plot,y1_plot),
    (x2_plot,y1_plot),
    (x2_plot,y2_plot),
    (x1_plot,y2_plot),
    (x1_plot,y1_plot)
])
metadata = {
    "dimensions" : {"height": height, "width": width}
}
field_points = {
    "agency_name_and_address": agency_name_and_address_mask
}

draw_masks(image, field_points, width, height)



# %%
from img2table import document
img = document.Image(src="output.png")
# Table identification
img_tables = img.extract_tables()
# %%
draw_bounding_box_table(image, img_tables, width, height)
# %%
def print_closest_blocks(blocks):
    
    id_visited_dict = {}
    id_block_dict = {block['Id']:block for block in blocks if block['BlockType'] == 'LINE'}
    
    data = []
    for block in blocks:
        if block['BlockType'] == 'LINE':
            data.append({
                "X":block['Geometry']['Polygon'][0]['X'],
                "Y":block['Geometry']['Polygon'][0]['Y'],
                "Id":block['Id']
            })
            id_visited_dict[block['Id']] = False

    df = pd.DataFrame(data)

    tree = skneighbors.BallTree(df[['X','Y']], leaf_size=2)  
    dist, ind = tree.query(df[['X','Y']][:1], k=len(df)) 

    select_index = df['Y'].idxmin()

    while(all(id_visited_dict.values()) == False):

        tree_search_distances, tree_search_indices = tree.query(df[['X','Y']][:select_index+1], k=len(df)) 
        for ind in tree_search_indices[0]:
            block_visited = id_visited_dict[df.iloc[ind]['Id']]

            if id_visited_dict[df.iloc[ind]['Id']] == False:
                id_visited_dict[df.iloc[ind]['Id']] = True
                print(id_block_dict[df.iloc[ind]['Id']]['Text'], '\n')

                break

        select_index = ind
              
print_closest_blocks(blocks_data)

# %%
import cv2
import numpy as np

# Read the image
image = cv2.imread('partial_image2.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, 10, 600)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area or other criteria
table_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 2000:  # You might need to adjust this threshold based on your image
        table_contours.append(contour)

# Draw bounding boxes around detected contours
for contour in table_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected table boundaries
# cv2.imshow('Table Boundaries', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('saved_image.jpg', image)

# %%
# %%
import cv2
import numpy as np
# from google.colab.patches import cv2_imshow
# !wget  https://i.stack.imgur.com/sDQLM.png
#read image 
image = cv2.imread( "partial_image2.png")

#convert to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#performing binary thresholding
kernel_size = 3
ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)  

#finding contours 
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#drawing Contours
radius =2
color = (30,255,50)
cv2.drawContours(image, cnts, -1,color , radius)
# cv2.imshow(image) commented as colab don't support cv2.imshow()
cv2.imwrite('saved_image_line.jpg', image)
# %%
import numpy as np
import cv2

gray = cv2.imread("partial_image2.png")
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imwrite('edges-50-150.jpg',edges)
minLineLength=100
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=700,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

a,b,c = lines.shape
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
cv2.imwrite('saved_image_line3.jpg',gray)

# %%
