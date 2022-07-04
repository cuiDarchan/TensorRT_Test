# onnxruntime_test.py
import onnxruntime as ort
import numpy as np
import cv2
from export_onnx import preprocessing

image_path = 'test.jpg'
ort_session = ort.InferenceSession("resnet50_wSoftmax.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']) # 创建一个推理session

img = cv2.imread(image_path)
input_img = preprocessing(img)
input_img = input_img.numpy()

pred = ort_session.run(None, { 'image' : input_img } )[0][0]
max_idx = np.argmax(pred)
print(f"test_image: {image_path}, max_idx: {max_idx}, probability: {pred[max_idx]}")
