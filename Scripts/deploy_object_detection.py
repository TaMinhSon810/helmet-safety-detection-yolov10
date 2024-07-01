import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLOv10
import locale


locale.getpreferredencoding = lambda: "UTF-8"

# Đường dẫn tới mô hình đã huấn luyện
TRAINED_MODEL_PATH = './yolov10/best.pt'

# Tải mô hình
model = YOLOv10(TRAINED_MODEL_PATH)


def process_image(image):
    CONF_THRESHOLD = 0.5
    IMG_SIZE = 640
    results = model.predict(source=image,
                            imgsz=IMG_SIZE,
                            conf=CONF_THRESHOLD)
    annotated_img = results[0].plot()
    return annotated_img


def main():
    st.title("Helmet Safety Detection")
    file = st.file_uploader("Upload Images", type=['JPG', 'PNG', 'JPEG'])
    if file is not None:
        st.image(file, caption="Uploaded Image")

        image = Image.open(file)
        image = np.array(image)
        processed_image = process_image(image)
        st.image(processed_image, caption="Processed Image")


if __name__ == "__main__":
    main()
