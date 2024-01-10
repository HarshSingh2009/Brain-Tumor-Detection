from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class predictPipeline():
    def __init__(self) -> None:
        self.brain_tumor_detect_model = YOLO('tumor_detector_model_YOLOv8.pt')

    def preprocess_img(self, img_path: str):
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)

        return img_array
    
    def detect_brain_tumors(self, preprocessed_img: np.array):
        detections = self.brain_tumor_detect_model(preprocessed_img)[0]
        brain_tumor_detections = []
        for brain_tumor in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = brain_tumor
            brain_tumor_detections.append([int(x1), int(y1), int(x2), int(y2), score])
        
        return brain_tumor_detections
    
    def drawDetections2Image(self, preprocessed_img: np.array, detections):
        img = np.array(preprocessed_img, dtype='uint8')
        for brain_tumor in detections:
            x1, y1, x2, y2, score = brain_tumor
            cv.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
            cv.putText(img, text=f'{round(score, 2)*100}%', org=(x1, y1-2), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), lineType=cv.LINE_AA)
        img_detections = np.array(img) 
        return img_detections
