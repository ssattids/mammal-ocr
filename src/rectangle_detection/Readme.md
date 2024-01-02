# TO PREDICT ON A NEW ACCORD FORMS
1. Modify the filepath `input_pdf` variable in `src/rectangle_detection/extract_key_value_pairs.py`
2. Run the file

# TO TRAIN A NEW ACCOR FORM
1. upload_rectangles_to_labelstudio.py - identify rectangles and prepare to upload annotations to label studio
   1. add images to ./data/not_filled_images
   2. add json to ./data/labelstudio_upload folder
2. get_non_filled_form_blocks.py - identify the keys in the text using OCR and it's coordinates
   1. add json to ./data/not_filled_ocr_blocks
3. assemble_label_studio_anns.py - assemble the data to identify the coordinates of the feilds
   1. add downloaded json from labelstudio into ./data/labelstudio_download.json
   2. add all the relevant data to identify fields in ./data/fields_data
4. For a filled form create it's images and use extract the key value fields
   1. add filled form to ./data/filled_images
