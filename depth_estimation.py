import urllib.request
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, render_template, Response
import time
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load MiDaS model for depth estimation
try:
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
except Exception as e:
    logging.error("Error loading MiDaS model: %s", e)
    raise e

midas.to('cpu')
midas.eval()

# Preprocessing for MiDaS
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load YOLOv8 model
from ultralytics import YOLO
model_yolo = YOLO('yolov8n.pt')

# Flask App
app = Flask(__name__)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Camera not opened. Check your camera connection!")
    raise Exception("Camera not opened.")

def generate_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame from camera.")
            continue

        # Object Detection
        try:
            results = model_yolo.predict(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2, conf, cls = box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item(), box.conf[0].item(), box.cls[0].item()
                    label = f"{model_yolo.names[int(cls)]}: {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            logging.error("YOLO prediction error: %s", e)

        # Depth Estimation
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            input_tensor = transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                depth_map = midas(input_tensor).squeeze().cpu().numpy()

            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

            # Resize depth map to match the frame size before concatenation
            depth_colormap = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))

            combined = cv2.hconcat([frame, depth_colormap])

        except Exception as e:
            logging.error("Depth estimation error: %s", e)
            combined = frame

        ret, buffer = cv2.imencode('.jpg', combined)
        if not ret:
            logging.warning("Failed to encode frame.")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/distance')
def distance():
    ret, frame = cap.read()
    if not ret:
        return {"error": "Frame capture failed"}, 500

    # Depth Estimation
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        input_tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            depth_map = midas(input_tensor).squeeze().cpu().numpy()

        # Calculate average distance in a central region of interest (ROI)
        h, w = depth_map.shape
        roi = depth_map[h//3:2*h//3, w//3:2*w//3]
        avg_distance = np.mean(roi)

        return {"average_distance": float(avg_distance)}

    except Exception as e:
        logging.error("Depth estimation error in /distance: %s", e)
        return {"error": "Depth estimation failed"}, 500



if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        cap.release()

