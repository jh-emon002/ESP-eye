import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

url = "http://192.168.43.26:81/stream"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Cannot open ESP32-CAM stream")

model = MobileNetV2(weights="imagenet")

frame_count = 0
text = "Initializing..."

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    if frame_count % 5 == 0:
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype("float32"))

        input_tensor = np.expand_dims(img, axis=0)
        preds = model.predict(input_tensor, verbose=0)

        _, description, confidence = decode_predictions(preds, top=1)[0][0]
        text = f"{description}: {confidence*100:.1f}%"

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("ESP32-CAM + MobileNetV2", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()