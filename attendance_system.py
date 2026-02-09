import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import winsound
import threading
import smtplib
from email.message import EmailMessage
import pyttsx3
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION
# ==========================================
SENDER_EMAIL = "saxhin0708@gmail.com"
SENDER_PASSWORD = "ikxp qdtd ufgg pbpk"   
TEACHER_EMAIL = "saxhin0708@gmail.com" #updates of  Attendance Of Todays 
SECURITY_EMAIL = "saxhin0708@gmail.com"   # Email to alert about intruders

class SmartAttendanceSystem:
    def __init__(self):
        self.encodings_path = "face_encodings_arcface.pkl"
        self.attendance_file = "attendance.csv"
        self.student_details_file = "students.csv"
        self.intruder_folder = "Intruders"
        
        # Settings
        self.confidence_threshold = 0.6  # Higher threshold for Security
        self.intruder_threshold = 30     # Frames (approx 2 seconds) to trigger alert
        self.cooldown_timer = 0          # Prevent spamming emails
        
        # Data Structures
        self.student_db = {} 
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_roll_nos = []
        self.present_students = set() 
        
        # --- GPU INITIALIZATION ---
        print("[INFO] Initializing High-Performance Security Engine...")
        # Using Large Model for maximum accuracy
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] Security Engine Loaded.")
        
        # Voice Engine
        self.engine = pyttsx3.init()
        self.spoken_ids = set()
        
        # Load Data
        self.load_student_details()
        self.load_encodings()
        self.init_attendance_csv()
        
        # Ensure intruder folder exists
        if not os.path.exists(self.intruder_folder):
            os.makedirs(self.intruder_folder)

    def load_student_details(self):
        if not os.path.exists(self.student_details_file):
            print(f"[ERROR] '{self.student_details_file}' not found!")
            exit()
        try:
            df = pd.read_csv(self.student_details_file, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            for index, row in df.iterrows():
                roll = row['Roll No'].strip()
                self.student_db[roll] = {
                    'name': row['Name'],
                    'email': row['Email']
                }
            print(f"[INFO] Loaded {len(self.student_db)} students.")
        except Exception as e:
            print(f"[ERROR] Reading CSV failed: {e}")
            exit()

    def load_encodings(self):
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
                self.known_roll_nos = data.get('roll_nos', [])
        else:
            print("[ERROR] Encodings missing! Run train_faces.py first.")
            exit()

    def init_attendance_csv(self):
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Roll No', 'Name', 'Status', 'Date', 'Time', 'Accuracy'])
            df.to_csv(self.attendance_file, index=False)

    def speak(self, text):
        def _speak():
            try:
                # ⚡ FIX: Create a fresh engine instance for every command
                # This prevents "driver jamming" when two people enter at once.
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[VOICE ERROR] {e}")
        
        threading.Thread(target=_speak, daemon=True).start()

    def trigger_intruder_alert(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.intruder_folder}/Intruder_{timestamp}.jpg"
        
        # 1. Save the Photo locally
        cv2.imwrite(filename, frame)
        print(f"[SECURITY] 🚨 Intruder Detected! Photo saved: {filename}")
        
        # 2. Play Alarm & Voice
        def play_alarm():
            try:
                winsound.Beep(1000, 1000) # Loud BEEP
                self.speak("Security Alert. Unauthorized person detected.")
            except: pass
        threading.Thread(target=play_alarm).start()
        
        # 3. Send Email WITH PHOTO ATTACHMENT
        def _send_email_with_photo():
            if not SENDER_PASSWORD: return
            
            msg = EmailMessage()
            msg['Subject'] = f"🚨 INTRUDER ALERT: {timestamp}"
            msg['From'] = SENDER_EMAIL
            msg['To'] = SECURITY_EMAIL
            msg.set_content(f"SECURITY ALERT!\n\nAn unknown person was detected at {datetime.now()}.\n\nSee the attached photo of the intruder.")

            # --- ATTACHMENT LOGIC ---
            try:
                with open(filename, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(filename)
                
                # Add the image to the email
                msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)
                
                # Send
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
                server.quit()
                print(f"[EMAIL SENT] 📸 Photo sent to Admin successfully.")
                
            except Exception as e:
                print(f"[EMAIL ERROR] Could not send photo: {e}")

        threading.Thread(target=_send_email_with_photo).start()
        
    def send_absent_email(self, roll, details):
        if not SENDER_PASSWORD: return
        msg = EmailMessage()
        msg.set_content(f"Dear Parent,\n\nStudent: {details['name']} ({roll})\nStatus: ABSENT\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\nPlease contact the college.\n\nRegards,\nSmart Attendance System")
        msg['Subject'] = f"Absent Alert: {details['name']}"
        msg['From'] = SENDER_EMAIL
        msg['To'] = details['email']
        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()
        except: pass

    def send_teacher_report(self, present_list, absent_list):
        if not TEACHER_EMAIL: return
        today_str = datetime.now().strftime("%Y-%m-%d")
        body = f"Daily Report ({today_str}):\nPresent: {len(present_list)}\nAbsent: {len(absent_list)}"
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = f"Attendance Report: {today_str}"
        msg['From'] = SENDER_EMAIL
        msg['To'] = TEACHER_EMAIL
        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"[REPORT SENT] 📄 Daily Summary sent.")
        except: pass

    def process_absentees(self):
        print("\n[INFO] SESSION ENDED. GENERATING REPORTS...")
        all_rolls = set(self.student_db.keys())
        present_rolls = self.present_students
        absent_rolls = all_rolls - present_rolls
        
        self.send_teacher_report(present_rolls, absent_rolls)

        if not absent_rolls: return

        absent_data = []
        for roll in absent_rolls:
            details = self.student_db.get(roll)
            if details:
                absent_data.append({
                    'Roll No': roll, 'Name': details['name'], 'Status': 'Absent', 
                    'Date': datetime.now().strftime("%Y-%m-%d"), 'Time': 'END', 'Accuracy': 'N/A'
                })
                threading.Thread(target=self.send_absent_email, args=(roll, details)).start()

        if absent_data:
            df = pd.DataFrame(absent_data)
            df.to_csv(self.attendance_file, mode='a', header=False, index=False)

    def mark_present(self, roll, ai_name, score):
        csv_name = self.student_db.get(roll, {}).get('name', ai_name)
        today = datetime.now().strftime("%Y-%m-%d")
        
        if roll not in self.present_students:
            score_percent = f"{int(score * 100)}%"
            print(f"[PRESENT] ✅ {csv_name} (Confidence: {score_percent})")
            
            new_entry = pd.DataFrame({
                'Roll No': [roll], 'Name': [csv_name], 'Status': ['Present'], 
                'Date': [today], 'Time': [datetime.now().strftime("%H:%M:%S")], 'Accuracy': [score_percent]
            })
            new_entry.to_csv(self.attendance_file, mode='a', header=False, index=False)
            self.present_students.add(roll)
            if roll not in self.spoken_ids:
                self.speak(f"Welcome {csv_name}")
                self.spoken_ids.add(roll)

    def cosine_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def run(self):
        print("[INFO] Starting Security Camera... Press 'q' to End Session.")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        frame_count = 0
        skip_frames = 2 
        current_faces = [] 
        
        # Intruder Variables
        unknown_consecutive_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            # --- PHASE 1: GPU DETECTION ---
            if frame_count % skip_frames == 0:
                try:
                    faces = self.app.get(frame)
                    current_faces = [] 
                    has_unknown = False
                    
                    for face in faces:
                        norm_emb = face.embedding / np.linalg.norm(face.embedding)
                        best_sim = 0
                        best_idx = -1
                        
                        for idx, known_emb in enumerate(self.known_face_encodings):
                            sim = self.cosine_similarity(norm_emb, known_emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_idx = idx
                        
                        name = "Unknown"
                        color = (0, 0, 255)
                        score_text = "0%"
                        
                        # High Security Threshold (0.6)
                        if best_sim >= self.confidence_threshold:
                            roll = self.known_roll_nos[best_idx]
                            name = self.student_db.get(roll, {}).get('name', self.known_face_names[best_idx])
                            color = (0, 255, 0)
                            self.mark_present(roll, name, best_sim)
                            display_score = min(99, int((best_sim + 0.2) * 100))
                            score_text = f"{display_score}%"
                        else:
                            # Flag that we found an intruder
                            has_unknown = True

                        current_faces.append((face.bbox.astype(int), name, color, score_text))
                    
                    # --- SECURITY LOGIC ---
                    if has_unknown:
                        unknown_consecutive_frames += 1
                    else:
                        unknown_consecutive_frames = 0 # Reset if safe
                    
                    # Decrease cooldown
                    if self.cooldown_timer > 0:
                        self.cooldown_timer -= 1

                    # Trigger Alert if Unknown for 30 frames (approx 2-3 secs)
                    if unknown_consecutive_frames > self.intruder_threshold and self.cooldown_timer == 0:
                        self.trigger_intruder_alert(frame)
                        self.cooldown_timer = 150 # Wait 150 frames (10 secs) before next alert
                        unknown_consecutive_frames = 0

                except: pass

            # --- PHASE 2: DRAWING ---
            for (bbox, name, color, score) in current_faces:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                # Show Alert on Screen if Unknown
                if name == "Unknown":
                    cv2.putText(frame, "WARNING: INTRUDER", (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"{name} {score}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('RTX 3050 Security', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.process_absentees()
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = SmartAttendanceSystem()
    system.run()