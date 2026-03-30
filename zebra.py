import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image

VIDEO_PATH = "edit.mp4"
MODEL_PATH = "yolov8n.pt"
FINAL_RESULTS = "final_results"

RESIZE_W, RESIZE_H = 960, 540
CONF_THRESHOLD = 0.4

# Zebra crossing polygon
bottom_plane = np.array([
    (87, 283),
    (700, 289),
    (658, 383),
    (4, 362),
], dtype=np.int32)

st.set_page_config(layout="wide")

st.title("🚦 Traffic Monitoring System")

frame_window = st.empty()

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# ---------------- VIDEO PLAYER ----------------
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

    # Draw zebra crossing
    cv2.polylines(frame, [bottom_plane], True, (0,0,255), 2)

    results = model(frame, conf=CONF_THRESHOLD)[0]

    if results.boxes is not None:
        for box in results.boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]

            if label not in ["car","motorcycle","bus","truck"]:
                continue

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_window.image(frame, channels="RGB")

cap.release()

st.success("✅ Video Completed")

st.divider()

# ---------------- VIOLATION DASHBOARD ----------------

st.header("Violations Recorded")

if os.path.exists(FINAL_RESULTS):

    folders = sorted(os.listdir(FINAL_RESULTS))

    if len(folders) == 0:
        st.info("No violations detected.")

    for folder in folders:

        folder_path = os.path.join(FINAL_RESULTS, folder)

        plate = folder.split("_")[-1]

        violated_frame = os.path.join(folder_path,"violated_frame.jpg")
        plate_img = os.path.join(folder_path,"plate.jpg")

        st.markdown("---")

        col1,col2,col3 = st.columns([1,2,2])

        with col1:
            st.markdown(f"**Vehicle**")
            st.write(folder)

            st.markdown("**Plate Number**")
            st.success(plate)

        with col2:
            st.markdown("**Violation Frame**")
            if os.path.exists(violated_frame):
                st.image(Image.open(violated_frame), use_container_width=True)

        with col3:
            st.markdown("**Detected Plate**")
            if os.path.exists(plate_img):
                st.image(Image.open(plate_img), use_container_width=True)