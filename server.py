# server.py
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from insightface.app import FaceAnalysis
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json

# --- CONFIG ---
app = FastAPI()

# Allow Lovable to talk to this computer
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REUSE YOUR BACKEND LOGIC HERE ---
# (Simplified for brevity - assumes you have your 'dataset' and 'encodings' ready)
class AttendanceSystem:
    def __init__(self):
        print("[INFO] Loading AI Engine...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.known_encodings = []
        self.known_names = []
        if os.path.exists("face_encodings_arcface.pkl"):
            with open("face_encodings_arcface.pkl", "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']

        self.attendance_log = [] # Temporary in-memory log
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        success, frame = self.cap.read()
        if not success: return None

        # --- AI PROCESSING ---
        faces = self.app.get(frame)
        for face in faces:
            norm = face.embedding / np.linalg.norm(face.embedding)
            max_sim = 0
            name = "Unknown"
            color = (0, 0, 255)

            for i, k in enumerate(self.known_encodings):
                sim = np.dot(norm, k)
                if sim > max_sim: max_sim = sim; idx = i

            if max_sim > 0.4:
                name = self.known_names[idx]
                color = (0, 255, 0)
                self.log_attendance(name)

            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{name} {int(max_sim*100)}%", (box[0], box[1]-10), 0, 0.6, color, 2)

        return frame

    def log_attendance(self, name):
        # Simple deduplication
        if not any(d['name'] == name for d in self.attendance_log):
            self.attendance_log.append({
                "name": name,
                "time": datetime.now().strftime("%H:%M:%S"),
                "status": "Present"
            })

# Initialize System
system = AttendanceSystem()

# --- API ENDPOINTS (Lovable connects to these) ---

@app.get("/stats")
def get_stats():
    return {
        "total_students": len(system.known_names),
        "present_count": len(system.attendance_log),
        "attendance_list": system.attendance_log
    }

def generate_video():
    while True:
        frame = system.get_frame()
        if frame is None: break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)