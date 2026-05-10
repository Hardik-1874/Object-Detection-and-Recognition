from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import time
import threading
import base64
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.ops  # Ensure your torchvision is CUDA-enabled

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", logger=False, engineio_logger=False)
# ─── Force usage of the dedicated NVIDIA GPU ──────────────────────
# Even though your laptop has an integrated GPU, only CUDA-capable devices count.
# Your RTX 3050 is likely the only CUDA device and will be indexed as "cuda:0".
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
# model.to(device)

# ─── Load YOLOv8 Model on the specified GPU ─────────────────────────
model = YOLO("yolov8s.pt")
model.to(device)
print(f"[INFO] YOLOv8 model loaded on {device}!")

# ─── Set up IP Camera Feed ─────────────────────────────────────────────
CAMERA_URL = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("[ERROR] Failed to open the camera feed. Check CAMERA_URL or network.")

native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Native camera resolution: {native_width} x {native_height}")

# Shared data (thread-safe)
lock = threading.Lock()
current_frame = None
current_detections = []
selected_object_id = None

def yolo_inference_loop():
    """
    Continuously reads frames from the camera, runs YOLOv8 inference,
    draws bounding boxes on the frame, and emits the frames via Socket.IO.
    """
    global cap, current_frame, current_detections

    # Use the native resolution for display.
    output_width, output_height = native_width, native_height

    # For inference, use a fixed width (e.g., 640) and compute the height to preserve aspect ratio.
    infer_width = 320
    infer_height = int(infer_width * output_height / output_width)

    # Scaling factors to convert inference-box coordinates back to the native resolution.
    scale_x = output_width / infer_width
    scale_y = output_height / infer_height

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        # Resize the frame to the native resolution.
        frame = cv2.resize(frame, (output_width, output_height))

        # Create a smaller version for inference.
        inference_frame = cv2.resize(frame, (infer_width, infer_height))

        # Run YOLO inference on the smaller image.
        results = model.predict(inference_frame, conf=0.5, verbose=False, imgsz=320)
        detections = []

        if results and len(results) > 0:
            for box in results[0].boxes:
                # Get coordinates on the inference image.
                x1, y1, x2, y2 = box.xyxy[0]
                # Scale coordinates back to the native resolution.
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_name": model.names[cls],
                    "conf": conf
                })

            # Draw bounding boxes and labels on the native-resolution frame.
            for det in detections:
                cv2.rectangle(frame, (det["x1"], det["y1"]),
                              (det["x2"], det["y2"]), (0, 255, 0), 2)
                label = f'{det["class_name"]} ({det["conf"] * 100:.1f}%)'
                cv2.putText(frame, label, (det["x1"], det["y1"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with lock:
            current_frame = frame.copy()
            current_detections = detections

        # Encode the frame as JPEG and then to base64.
        _, buffer = cv2.imencode('.jpg', frame)
        b64_frame = base64.b64encode(buffer).decode('utf-8')

        socketio.emit('video_frame', {'data': b64_frame})
        socketio.emit('detections', detections)

        time.sleep(0.05)

@app.route('/')
def index():
    """Serve the main HTML page that connects to Socket.IO."""
    return render_template('index.html', 
                           video_width=native_width, 
                           video_height=native_height)

@app.route('/select_object', methods=['POST'])
def select_object():
    """Set the selected object based on user click coordinates."""
    global selected_object_id
    data = request.json
    click_x = data.get('click_x')
    click_y = data.get('click_y')

    with lock:
        found_idx = None
        for idx, det in enumerate(current_detections):
            if (det["x1"] <= click_x <= det["x2"]) and (det["y1"] <= click_y <= det["y2"]):
                found_idx = idx
                break
        selected_object_id = found_idx

    return jsonify({"selected_object_id": selected_object_id})

if __name__ == '__main__': 
    socketio.start_background_task(yolo_inference_loop)
# CORRECT
socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)