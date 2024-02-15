import numpy as np
import pandas as pd
import easyocr
import streamlit as st
from PIL import Image
import cv2
import base64


# Function to add app background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(f"""<style>.stApp {{background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover}}</style>""", unsafe_allow_html=True)


def display_ocr_image(img, results):
    img_np = np.array(img)

    for detection in results:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        font = cv2.FONT_HERSHEY_COMPLEX

        # text_size = cv2.getTextSize(text, font, 1, 2)[0]

        # text_position = (top_left[0] + 10, top_left[1] + text_size[1] + 10)  # Adjust the position as needed

        cv2.rectangle(img_np, top_left, bottom_right, (0, 255, 0), 5)
        cv2.putText(img_np, text, top_left, font, 1, (125, 29, 241), 2, cv2.LINE_AA)

    # Display the image with bounding boxes and text
    st.image(img_np, channels="BGR", use_column_width=True)


def extracted_text(col):
    return " , ".join(img_df[col])


# Streamlit app
st.markdown("""
    <svg width="600" height="100">
        <text x="50%" y="50%" font-family="monospace" font-size="42px" fill="Turquoise" text-anchor="middle" stroke="white"
         stroke-width="0.3" stroke-linejoin="round">📃 ScanMaster OCR 📃
        </text>
    </svg>
""", unsafe_allow_html=True)

add_bg_from_local('background.jpg')

file = st.file_uploader(label= "Upload Image Here (png/jpg/jpeg) : ", type=['png', 'jpg', 'jpeg'])

if file is not None:
    image = Image.open(file)
    st.image(image)

    reader = easyocr.Reader(['en', 'hi'], gpu=False)
    results = reader.readtext(np.array(image))

    img_df = pd.DataFrame(results, columns=['bbox', 'Predicted Text', 'Prediction Confidence'])

    # Getting all text extracted
    text_combined = extracted_text(col='Predicted Text')
    st.write("Text Generated :- ", text_combined)

    # Printing results in tabular form
    st.write("Table Showing Predicted Text and Prediction Confidence : ")
    st.table(img_df.iloc[:, 1:])

    # getting final image with drawing annotations
    display_ocr_image(image, results)

else:
    st.warning("!! Please Upload your image !!")



