import os                           # Used to work with folders and file paths
import cv2                          #OpenCV library for reading images
import numpy as np                  # NumPy for numerical operations (arrays, vectors)
import pickle                       # Used to save/load trained data
from insightface.app import FaceAnalysis   # ArcFace face detection & embedding model

# ================= CONFIG =================
DATASET_DIR = "dataset" 
#This  Folder containing subfolfers student.
# Each student folder contains multiple face images

OUTPUT_FILE = "face_encodings_arcface.pkl"
# This file will store all trained face embeddings (brain of system)

# ==========================================


# ===========================
# DISPLAY PROGRAM HEADER
# ===========================
print("\n")
print("=" * 60)
print("   ARCFACE HIGH-ACCURACY TRAINING SYSTEM")
print("=" * 60)

# 1. Initialize ArcFace Model (The "Brain")
print("[INFO] Initializing ArcFace model (this might take a moment)...")
# FaceAnalysis loads:
# - Face detector
# - ArcFace recognition model
# 'buffalo_l' is the most accurate pre-trained model
# We MUST use 'buffalo_s' to match the attendance system
# CUDAExecutionProvider = GPU (if available)
# CPUExecutionProvider = fallback to CPU

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640)) # det_size → face detection resolution
print("[INFO] Model Loaded Successfully")


# ===========================
# DATA STORAGE LISTS
# ===========================

known_face_encodings = []        # Stores 512-D face vectors
known_face_names = []            # Stores student names
known_face_rolls = []             # Stores student roll numbers


# 2. Check Dataset
# If dataset folder does not exist → stop program
if not os.path.exists(DATASET_DIR):
    print(f"[ERROR] Dataset folder not found: {DATASET_DIR}")
    exit()
# Get list of student folders
students = [s for s in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, s))]
print(f"\n[INFO] Found {len(students)} students to train")

# If no students found → stop
if len(students) == 0:
    print("[ERROR] No student folders found in dataset_cropped")
    exit()
    

# 3. Start Training
for student in students:
    student_path = os.path.join(DATASET_DIR, student)   # Path of each student folder

    # -------- Parse folder name: Name_RollNo --------
    # Example: "Sachin_25CS094" -> Name: Sachin, Roll: 25CS094
    if "_" in student:
        parts = student.split("_", 1)
        name = parts[0]
        roll_no = parts[1]
    else:
        name = student
        roll_no = "000"

    print(f"\n[PROCESSING] Student: {name} (Roll: {roll_no})")

    # Get all image files in student folder
    images = [img for img in os.listdir(student_path) if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    for image_name in images:
        image_path = os.path.join(student_path, image_name)
        
        # Read image using OpenCV
        img = cv2.imread(image_path)   
        if img is None:
            print(f"  [WARNING] Could not read image: {image_name}")
            continue

        # -------- ARCFACE DETECTION --------
        # ArcFace detects faces and generates the 512-D embedding in one go
        faces = app.get(img)

        if len(faces) == 0:    # If no face found → skip image
            print(f"  [SKIP] No face detected in {image_name}")
            continue
        
        # Take the largest face found in the image
        # Sort by size (width * height) to ensure we get the main subject
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        
        # Get the 512-dimensional embedding
        embedding = faces[0].embedding
        
        # Normalize the embedding (Critical for Cosine Similarity)
        embedding = embedding / np.linalg.norm(embedding)

        known_face_encodings.append(embedding)
        known_face_names.append(name)
        known_face_rolls.append(roll_no)
        print(f"  [OK] Encoded {image_name}")

# 4. Save to Pickle
if len(known_face_encodings) == 0:
    print("\n[ERROR] No faces were successfully encoded. Check your images.")
    exit()

data = {
    "encodings": known_face_encodings,
    "names": known_face_names,
    "roll_nos": known_face_rolls
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("\n" + "=" * 60)
print("[SUCCESS] TRAINING COMPLETE")
print(f"[SAVED] {OUTPUT_FILE}")
print(f"[INFO] Total Faces Encoded: {len(known_face_encodings)}")
print(f"[INFO] Embedding Dimension: {len(known_face_encodings[0])} (Should be 512)")
print("=" * 60)