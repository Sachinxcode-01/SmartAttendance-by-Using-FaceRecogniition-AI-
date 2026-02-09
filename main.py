import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import os
import shutil
import pickle
import threading
import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import pyttsx3
import winsound
import smtplib
from email.message import EmailMessage
from insightface.app import FaceAnalysis
import random
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Email Config
SENDER_EMAIL = "saxhin0708@gmail.com"
SENDER_PASSWORD = "ikxp qdtd ufgg pbpk"   
TEACHER_EMAIL = "saxhin0708@gmail.com"
SECURITY_EMAIL = "saxhin0708@gmail.com"

# Login Credentials
ADMIN_USER = "admin"
ADMIN_PASS = "password123"

# Paths
DATASET_DIR = "dataset"
ENCODINGS_FILE = "face_encodings_arcface.pkl"
ATTENDANCE_FILE = "attendance.csv"
STUDENT_FILE = "student.csv"
INTRUDER_FOLDER = "Intruders"

# AI Settings
CONFIDENCE_THRESHOLD = 0.50
INTRUDER_THRESHOLD = 30  # Frames

# Colors
COLOR_BG_DARK = "#050A15"       
COLOR_BG_LIGHT = "#101929"      
COLOR_ACCENT_CYAN = "#00E0FF"   
COLOR_ACCENT_PURPLE = "#9D00FF" 
COLOR_GLASS = "#1A2235"         
COLOR_TEXT_GRAY = "#8A95A5"
COLOR_DANGER = "#ef4444"
COLOR_SUCCESS = "#22c55e"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# ==========================================
# 2. BACKEND LOGIC (The Brain)
# ==========================================
class AttendanceBackend:
    def __init__(self):
        # Create Directories
        for folder in [DATASET_DIR, INTRUDER_FOLDER]:
            if not os.path.exists(folder): os.makedirs(folder)
            
        # Init Files
        if not os.path.exists(STUDENT_FILE):
            pd.DataFrame(columns=['Roll No', 'Name', 'Email', 'Gender']).to_csv(STUDENT_FILE, index=False)
        if not os.path.exists(ATTENDANCE_FILE):
            pd.DataFrame(columns=['Roll No', 'Name', 'Status', 'Date', 'Time', 'Accuracy']).to_csv(ATTENDANCE_FILE, index=False)

        # Load AI Engine
        print("[INFO] Initializing GPU Engine (buffalo_l)...")
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except:
            print("[WARNING] CUDA not found. Falling back to CPU.")
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load Data
        self.known_encodings = []
        self.known_names = []
        self.known_rolls = []
        self.load_encodings()
        
        # State
        self.student_db = self.load_student_db()
        self.present_students = set()
        self.recent_alerts = []
        self.spoken_ids = set()
        
    def load_student_db(self):
        try:
            df = pd.read_csv(STUDENT_FILE, dtype=str)
            db = {}
            for _, row in df.iterrows():
                db[row['Roll No']] = {'name': row['Name'], 'email': row['Email']}
            return db
        except: return {}

    def load_encodings(self):
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
                self.known_rolls = data.get('roll_nos', [])
            print(f"[INFO] Loaded {len(self.known_encodings)} trained faces.")

    def train_system(self):
        """Re-scans the dataset folder and updates encodings (Logic from train_faces.py)."""
        print("[TRAINING] Starting training process...")
        known_encs = []
        known_names = []
        known_rolls = []
        
        students = [s for s in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, s))]
        
        for student in students:
            # Parse Name_Roll
            if "_" in student:
                name, roll = student.split("_", 1)
            else:
                name, roll = student, "Unknown"
                
            path = os.path.join(DATASET_DIR, student)
            images = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_name in images:
                img = cv2.imread(os.path.join(path, img_name))
                if img is None: continue
                
                faces = self.app.get(img)
                if len(faces) > 0:
                    # Get largest face
                    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                    emb = faces[0].embedding
                    emb = emb / np.linalg.norm(emb)
                    
                    known_encs.append(emb)
                    known_names.append(name)
                    known_rolls.append(roll)
        
        data = {"encodings": known_encs, "names": known_names, "roll_nos": known_rolls}
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        
        self.known_encodings = known_encs
        self.known_names = known_names
        self.known_rolls = known_rolls
        print("[TRAINING] Complete.")
        return len(known_encs)

    def mark_attendance(self, roll, name, acc):
        if roll not in self.present_students:
            now = datetime.now()
            score_str = f"{int(acc*100)}%"
            
            new_entry = pd.DataFrame({
                'Roll No': [roll], 'Name': [name], 'Status': ['Present'],
                'Date': [now.strftime("%Y-%m-%d")], 'Time': [now.strftime("%H:%M:%S")],
                'Accuracy': [score_str]
            })
            new_entry.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)
            
            self.present_students.add(roll)
            self.recent_alerts.insert(0, {"title": f"Entry: {name}", "time": "Just now", "type": "success"})
            
            if roll not in self.spoken_ids:
                self.speak_thread(f"Welcome {name}")
                self.spoken_ids.add(roll)
            return True
        return False

    def speak_thread(self, text):
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except: pass
        threading.Thread(target=_speak, daemon=True).start()

    def send_email(self, subject, body, attachment=None):
        def _send():
            if not SENDER_PASSWORD: return
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = SENDER_EMAIL
            msg['To'] = SECURITY_EMAIL
            msg.set_content(body)
            if attachment:
                try:
                    with open(attachment, 'rb') as f:
                        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=os.path.basename(attachment))
                except: pass
            try:
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
                server.quit()
            except Exception as e: print(e)
        threading.Thread(target=_send).start()

# ==========================================
# 3. UI COMPONENTS (Login & Dashboard)
# ==========================================

class LoginWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SECURITY LOGIN")
        self.geometry("400x500")
        self.configure(fg_color=COLOR_BG_DARK)
        self.resizable(False, False)

        ctk.CTkLabel(self, text="🔒 SYSTEM ACCESS", font=("Arial", 20, "bold"), text_color=COLOR_ACCENT_CYAN).pack(pady=(50, 10))
        ctk.CTkLabel(self, text="Please verify your identity", font=("Arial", 12), text_color=COLOR_TEXT_GRAY).pack(pady=(0, 40))

        self.user_entry = ctk.CTkEntry(self, placeholder_text="Username", width=250, height=40, corner_radius=20, fg_color=COLOR_GLASS, border_color=COLOR_BG_LIGHT)
        self.user_entry.pack(pady=10)

        self.pass_entry = ctk.CTkEntry(self, placeholder_text="Password", show="*", width=250, height=40, corner_radius=20, fg_color=COLOR_GLASS, border_color=COLOR_BG_LIGHT)
        self.pass_entry.pack(pady=10)

        self.error_label = ctk.CTkLabel(self, text="", text_color=COLOR_DANGER, font=("Arial", 12))
        self.error_label.pack(pady=5)

        ctk.CTkButton(self, text="LOGIN", width=250, height=40, corner_radius=20, fg_color=COLOR_ACCENT_CYAN, hover_color="#00b3cc", text_color="black", font=("Arial", 14, "bold"), command=self.check_login).pack(pady=20)

    def check_login(self):
        if self.user_entry.get() == ADMIN_USER and self.pass_entry.get() == ADMIN_PASS:
            self.destroy()
            app = AttendanceApp()
            app.mainloop()
        else:
            self.error_label.configure(text="ACCESS DENIED")
            winsound.Beep(500, 200)

class NeuralNetworkBackground(ctk.CTkCanvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, highlightthickness=0, **kwargs)
        self.configure(bg=COLOR_BG_DARK)
        self.num_particles = 30
        self.particles = [{"x": random.randint(0, 1300), "y": random.randint(0, 750), "vx": random.uniform(-0.5, 0.5), "vy": random.uniform(-0.5, 0.5)} for _ in range(self.num_particles)]
        self.animate()

    def animate(self):
        self.delete("all")
        for p in self.particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]
            if p["x"] < 0 or p["x"] > 1300: p["vx"] *= -1
            if p["y"] < 0 or p["y"] > 750: p["vy"] *= -1
            self.create_oval(p["x"]-2, p["y"]-2, p["x"]+2, p["y"]+2, fill="#1e293b", outline="")
        
        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                dist = math.hypot(p1["x"]-p2["x"], p1["y"]-p2["y"])
                if dist < 150:
                    c = f"{int(20+(1-dist/150)*40):02x}"
                    self.create_line(p1["x"], p1["y"], p2["x"], p2["y"], fill=f"#{c}{c}{c}", width=1)
        self.after(33, self.animate)

class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.backend = AttendanceBackend()
        
        self.title("AI ATTENDANCE PRO")
        self.geometry("1400x800")
        
        # Background
        self.bg = NeuralNetworkBackground(self, width=1400, height=800)
        self.bg.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.setup_layout()
        self.show_frame("Dashboard")
        
        # Camera
        self.cap = None
        self.camera_running = False
        self.start_camera()

    def setup_layout(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color=COLOR_GLASS)
        self.sidebar.place(x=0, y=0, relheight=1)
        
        ctk.CTkLabel(self.sidebar, text="🧠 AI VISION", font=("Arial", 24, "bold"), text_color=COLOR_ACCENT_CYAN).pack(pady=40)
        self.create_nav("Dashboard", lambda: self.show_frame("Dashboard"))
        self.create_nav("Manage Students", lambda: self.show_frame("Manage"))
        self.create_nav("Reports", lambda: self.show_frame("Reports"))
        ctk.CTkButton(self.sidebar, text="LOGOUT", fg_color=COLOR_DANGER, command=self.on_close).pack(side="bottom", pady=20)

        # Content Area
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.place(x=220, y=20, relwidth=0.82, relheight=0.95)
        
        self.frames = {}
        self.frames["Dashboard"] = self.create_dashboard()
        self.frames["Manage"] = self.create_manage()
        self.frames["Reports"] = self.create_reports()

    def create_nav(self, text, cmd):
        ctk.CTkButton(self.sidebar, text=text, height=45, fg_color="transparent", hover_color=COLOR_ACCENT_CYAN, anchor="w", command=cmd).pack(fill="x", padx=10, pady=5)

    def show_frame(self, name):
        for f in self.frames.values(): f.pack_forget()
        self.frames[name].pack(fill="both", expand=True)
        if name == "Reports": self.refresh_logs()

    # --- PAGES ---
    def create_dashboard(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        
        # Video Area
        self.video_label = ctk.CTkLabel(frame, text="", corner_radius=20)
        self.video_label.pack(side="left", padx=20, pady=20)
        
        # Right Panel
        panel = ctk.CTkFrame(frame, width=300, fg_color="transparent")
        panel.pack(side="right", fill="y", padx=20)
        
        self.stat_card = ctk.CTkFrame(panel, height=120, fg_color=COLOR_GLASS, border_color=COLOR_ACCENT_CYAN, border_width=1)
        self.stat_card.pack(fill="x", pady=20)
        self.lbl_total = ctk.CTkLabel(self.stat_card, text="TOTAL: 0", font=("Arial", 22, "bold"))
        self.lbl_total.pack(pady=20)
        
        ctk.CTkLabel(panel, text="LIVE ALERTS", font=("Arial", 16, "bold")).pack(pady=10)
        self.alert_box = ctk.CTkFrame(panel, fg_color="transparent")
        self.alert_box.pack(fill="both", expand=True)
        self.alert_widgets = []
        
        return frame

    def create_manage(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        ctk.CTkLabel(frame, text="ADD NEW STUDENT", font=("Arial", 24, "bold")).pack(pady=30)
        
        form = ctk.CTkFrame(frame, fg_color=COLOR_GLASS)
        form.pack(pady=20, padx=100, fill="x")
        
        self.entry_name = ctk.CTkEntry(form, placeholder_text="Full Name"); self.entry_name.pack(pady=10, padx=50, fill="x")
        self.entry_roll = ctk.CTkEntry(form, placeholder_text="Roll No"); self.entry_roll.pack(pady=10, padx=50, fill="x")
        self.entry_email = ctk.CTkEntry(form, placeholder_text="Email"); self.entry_email.pack(pady=10, padx=50, fill="x")
        
        btn_box = ctk.CTkFrame(frame, fg_color="transparent")
        btn_box.pack(pady=20)
        ctk.CTkButton(btn_box, text="📸 Capture Photos", command=self.capture_photos).pack(side="left", padx=20)
        ctk.CTkButton(btn_box, text="📂 Upload Photos", command=self.upload_photos).pack(side="left", padx=20)
        
        ctk.CTkButton(frame, text="🚀 SAVE & TRAIN SYSTEM", height=50, fg_color=COLOR_SUCCESS, command=self.save_train).pack(pady=30)
        return frame

    def create_reports(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        ctk.CTkLabel(frame, text="ATTENDANCE LOGS", font=("Arial", 24, "bold")).pack(pady=20)
        
        self.log_text = ctk.CTkTextbox(frame, width=800, height=400)
        self.log_text.pack(pady=10)
        
        ctk.CTkButton(frame, text="📊 Export to Excel", fg_color=COLOR_SUCCESS, command=self.export_excel).pack(pady=20)
        return frame

    # --- FUNCTIONALITY ---
    def start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_running = True
        self.frame_count = 0
        self.unknown_count = 0
        self.process_camera()

    def process_camera(self):
        if not self.camera_running: return
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            display = frame.copy()
            
            if self.frame_count % 2 == 0:
                try:
                    faces = self.backend.app.get(frame)
                    has_unknown = False
                    for face in faces:
                        norm = face.embedding / np.linalg.norm(face.embedding)
                        max_sim = 0; idx = -1
                        for i, k in enumerate(self.backend.known_encodings):
                            sim = np.dot(norm, k)
                            if sim > max_sim: max_sim = sim; idx = i
                        
                        bbox = face.bbox.astype(int)
                        if max_sim >= CONFIDENCE_THRESHOLD:
                            name = self.backend.known_names[idx]
                            roll = self.backend.known_rolls[idx]
                            self.backend.mark_attendance(roll, name, max_sim)
                            color = (0, 255, 0)
                        else:
                            name = "Unknown"; color = (0, 0, 255); has_unknown = True
                        
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(display, f"{name} {int(max_sim*100)}%", (bbox[0], bbox[1]-10), 0, 0.6, color, 2)
                    
                    if has_unknown: self.unknown_count += 1
                    else: self.unknown_count = 0
                    
                    if self.unknown_count > INTRUDER_THRESHOLD:
                        self.trigger_intruder(frame)
                        self.unknown_count = -100
                except: pass
            
            # Update Stats & UI
            self.lbl_total.configure(text=f"Present: {len(self.backend.present_students)}")
            if self.frame_count % 30 == 0: self.update_alerts_ui()

            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (700, 450))
            imgtk = ctk.CTkImage(light_image=Image.fromarray(img), size=(700, 450))
            self.video_label.configure(image=imgtk)
        
        self.after(10, self.process_camera)

    def trigger_intruder(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{INTRUDER_FOLDER}/Intruder_{ts}.jpg"
        cv2.imwrite(path, frame)
        winsound.Beep(1000, 500)
        self.backend.speak_thread("Warning. Intruder detected.")
        self.backend.send_email(f"🚨 Intruder Alert {ts}", "Unauthorized person detected.", path)
        self.backend.recent_alerts.insert(0, {"title": "INTRUDER DETECTED", "time": "Just now", "type": "danger"})

    def update_alerts_ui(self):
        for w in self.alert_widgets: w.destroy()
        self.alert_widgets = []
        for a in self.backend.recent_alerts[:4]:
            c = COLOR_DANGER if a['type'] == 'danger' else COLOR_SUCCESS
            f = ctk.CTkFrame(self.alert_box, height=50, fg_color=COLOR_GLASS, border_color=c, border_width=1)
            f.pack(fill="x", pady=5)
            ctk.CTkLabel(f, text=a['title'], font=("Arial", 12, "bold"), text_color="white").pack(side="left", padx=10)
            ctk.CTkLabel(f, text=a['time'], text_color="gray").pack(side="right", padx=10)
            self.alert_widgets.append(f)

    # --- MANAGEMENT ---
    def capture_photos(self):
        name = self.entry_name.get(); roll = self.entry_roll.get()
        if not name or not roll: return messagebox.showerror("Error", "Enter details first!")
        folder = f"{DATASET_DIR}/{name}_{roll}"
        if not os.path.exists(folder): os.makedirs(folder)
        
        self.camera_running = False; self.cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        count = 0
        while count < 20:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow("Space to Capture", frame)
            if cv2.waitKey(1) & 0xFF == 32:
                cv2.imwrite(f"{folder}/{count}.jpg", frame); count+=1; print(f"Captured {count}")
        cap.release(); cv2.destroyAllWindows()
        messagebox.showinfo("Done", "Captured 20 photos."); self.start_camera()

    def upload_photos(self):
        name = self.entry_name.get(); roll = self.entry_roll.get()
        if not name or not roll: return messagebox.showerror("Error", "Enter details first!")
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.png")])
        if files:
            folder = f"{DATASET_DIR}/{name}_{roll}"
            if not os.path.exists(folder): os.makedirs(folder)
            for i, f in enumerate(files): shutil.copy(f, f"{folder}/up_{i}.jpg")
            messagebox.showinfo("Done", f"Uploaded {len(files)} photos.")

    def save_train(self):
        name = self.entry_name.get(); roll = self.entry_roll.get(); email = self.entry_email.get()
        if name and roll:
            df = pd.DataFrame({'Roll No': [roll], 'Name': [name], 'Email': [email], 'Gender': ['N/A']})
            df.to_csv(STUDENT_FILE, mode='a', header=False, index=False)
            messagebox.showinfo("Training", "System is learning new faces... Wait.")
            self.update()
            c = self.backend.train_system()
            messagebox.showinfo("Success", f"Trained {c} faces!")

    def refresh_logs(self):
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f: self.log_text.insert("1.0", f.read())

    def export_excel(self):
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            f = f"Attendance_{datetime.now().strftime('%Y%m%d')}.xlsx"
            df.to_excel(f, index=False)
            messagebox.showinfo("Success", f"Saved {f}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def on_close(self):
        self.camera_running = False
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = LoginWindow()
    app.mainloop()