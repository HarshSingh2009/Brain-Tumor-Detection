import streamlit as st
from pipeline import predictPipeline


st.title('Brain Tumor detection')
st.write('Detects Tumors in a MRI scan of a Brain \nPowered by YOLOv8 Medium model')

st.write('')

detect_pipeline = predictPipeline()

st.info('Brain Tumor Detection model loaded successfully!')


st.warning('Please do not upload any image that is not of a MRI scan of a Brain image, else this model will be forced to predict randomly')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    with st.container():
        col1, col2 = st.columns([3, 3])
        col1.header('Input Image')
        col1.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        col1.text('')
        col1.text('')

        if st.button('Detect'):
            preprocess_img_array = detect_pipeline.preprocess_img(img_path=uploaded_file)
            tumor_detections = detect_pipeline.detect_brain_tumors(preprocessed_img=preprocess_img_array)
            detections_img = detect_pipeline.drawDetections2Image(preprocessed_img=preprocess_img_array, detections=tumor_detections)

            col2.header('Detections')
            col2.image(detections_img, caption='Predictions by model', use_column_width=True)

