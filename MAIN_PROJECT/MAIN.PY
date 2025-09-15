
# main.py
"""
PresenceAI - Attendance Backend (Enhanced with Better Liveness Detection)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import sqlite3
import os
import io
import json
import uuid
import secrets
import hashlib
import binascii
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, date, timedelta
import asyncio
import base64
import math
import pathlib

# Face libs (requires dlib, face_recognition, opencv installed)
import face_recognition
import numpy as np
import cv2

# Ensure paths are resolved relative to this file
BASE_DIR = pathlib.Path(__file__).resolve().parent

# Directories (use BASE_DIR so running from different cwd still works)
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
AVATARS_DIR = BASE_DIR / "avatars"
TEMP_DIR = BASE_DIR / "temp"
for d in (KNOWN_FACES_DIR, AVATARS_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

# App init
app = FastAPI(title="PresenceAI Attendance Backend", version="1.0.0")

# Mount static & avatars directories
try:
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
except Exception as e:
    print("[WARN] Could not mount /static:", e)

try:
    app.mount("/avatars_static", StaticFiles(directory=str(AVATARS_DIR)), name="avatars_static")
except Exception as e:
    print("[WARN] Could not mount /avatars_static:", e)

# CORS - for local dev allow everything
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB file (placed in BASE_DIR)
DB_PATH = str(BASE_DIR / "attendance.db")

# In-memory token store (demo)
TOKENS: Dict[str, str] = {}

# Known faces cache
known_encodings: List[np.ndarray] = []
known_names: List[str] = []

# ------------------------
# FIXED Enhanced Liveness Detection Module
# ------------------------

class LivenessDetector:
    def _init_(self):
        self.frame_buffer = deque(maxlen=10)  # Store last 10 frames for motion analysis
        self.blink_frames = deque(maxlen=5)   # Store frames for blink detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except Exception as e:
            print(f"[WARN] Could not load cascade classifiers: {e}")
            self.face_cascade = None
            self.eye_cascade = None
    
    def detect_motion(self, current_frame, previous_frame):
        """Detect motion between consecutive frames"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY) if len(previous_frame.shape) == 3 else previous_frame
            gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame
            
            # Ensure same dimensions
            if gray1.shape != gray2.shape:
                return False
            
            # Compute absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Apply threshold
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Count non-zero pixels (motion pixels)
            motion_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            motion_ratio = motion_pixels / total_pixels
            
            return motion_ratio > 0.003  # Lowered threshold for subtle motion
            
        except Exception as e:
            print(f"[WARN] Motion detection error: {e}")
            return False
    
    def check_image_quality_and_sharpness(self, image):
        """Check if image is too sharp/artificial (printed photo indicator)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate Laplacian variance (sharpness measure)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x*2 + grad_y*2)
            avg_gradient = np.mean(gradient_magnitude)
            
            # Check for over-sharpening (common in printed photos)
            if laplacian_var > 2000:  # Too sharp - likely printed/enhanced
                return True
                
            # Check for unnatural uniformity (printed photos often lack natural variation)
            pixel_std = np.std(gray)
            if pixel_std < 15:  # Too uniform
                return True
                
            # Check histogram for unnatural peaks
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_peaks = np.where(hist > np.max(hist) * 0.1)[0]
            if len(hist_peaks) < 8:  # Too few intensity levels
                return True
                
            return False
            
        except Exception as e:
            print(f"[WARN] Image quality check error: {e}")
            return False
    
    def check_skin_tone_analysis(self, face_region):
        """Analyze skin tone patterns to detect printed photos"""
        try:
            if len(face_region.shape) != 3:
                return False
                
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Analyze skin tone distribution
            h_channel = hsv[:,:,0]
            s_channel = hsv[:,:,1]
            
            # Check for unnatural skin tone uniformity
            h_std = np.std(h_channel)
            s_std = np.std(s_channel)
            
            # Real skin has natural variation, printed photos are often too uniform
            if h_std < 8 or s_std < 12:
                return True
                
            # Check for color quantization (printing artifacts)
            unique_colors = len(np.unique(face_region.reshape(-1, face_region.shape[2]), axis=0))
            total_pixels = face_region.shape[0] * face_region.shape[1]
            color_ratio = unique_colors / total_pixels
            
            if color_ratio < 0.3:  # Too few unique colors
                return True
                
            return False
            
        except Exception as e:
            print(f"[WARN] Skin tone analysis error: {e}")
            return False
    
    def check_texture_patterns(self, image):
        """Advanced texture analysis to detect printed material"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Resize for consistent analysis
            if gray.shape[0] > 150 or gray.shape[1] > 150:
                gray = cv2.resize(gray, (150, 150))
            
            # Local Binary Pattern analysis
            def local_binary_pattern(image, radius=1):
                rows, cols = image.shape
                lbp = np.zeros_like(image)
                
                for i in range(radius, rows - radius):
                    for j in range(radius, cols - radius):
                        center = image[i, j]
                        binary_string = ""
                        
                        # 8 surrounding pixels
                        neighbors = [
                            image[i-radius, j-radius], image[i-radius, j], image[i-radius, j+radius],
                            image[i, j+radius], image[i+radius, j+radius], image[i+radius, j],
                            image[i+radius, j-radius], image[i, j-radius]
                        ]
                        
                        for neighbor in neighbors:
                            binary_string += '1' if neighbor >= center else '0'
                        
                        lbp[i, j] = int(binary_string, 2)
                
                return lbp
            
            lbp = local_binary_pattern(gray)
            lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            
            # Analyze texture uniformity
            texture_uniformity = np.max(lbp_hist) / np.sum(lbp_hist)
            
            if texture_uniformity > 0.15:  # Too uniform texture pattern
                return True
                
            # Check for printing artifacts using frequency domain
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Look for artificial frequency patterns
            center_energy = np.mean(magnitude_spectrum[
                magnitude_spectrum.shape[0]//4:3*magnitude_spectrum.shape[0]//4,
                magnitude_spectrum.shape[1]//4:3*magnitude_spectrum.shape[1]//4
            ])
            
            edge_energy = np.mean([
                np.mean(magnitude_spectrum[:magnitude_spectrum.shape[0]//4, :]),
                np.mean(magnitude_spectrum[3*magnitude_spectrum.shape[0]//4:, :]),
                np.mean(magnitude_spectrum[:, :magnitude_spectrum.shape[1]//4]),
                np.mean(magnitude_spectrum[:, 3*magnitude_spectrum.shape[1]//4:])
            ])
            
            if center_energy / (edge_energy + 1e-6) > 2.5:  # Unnatural frequency distribution
                return True
                
            return False
            
        except Exception as e:
            print(f"[WARN] Texture pattern analysis error: {e}")
            return False
    
    def check_3d_face_structure(self, image):
        """Analyze face structure to detect flat/2D images"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            if self.face_cascade is None:
                return False
            
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                return False
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face_roi = gray[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return False
            
            # Analyze shadow gradients (3D faces have natural shadows)
            # Divide face into regions and check for natural lighting gradients
            h_third = h // 3
            w_third = w // 3
            
            regions = [
                face_roi[0:h_third, 0:w_third],          # Top-left
                face_roi[0:h_third, w_third:2*w_third],  # Top-center
                face_roi[0:h_third, 2*w_third:],         # Top-right
                face_roi[h_third:2*h_third, 0:w_third],  # Mid-left
                face_roi[h_third:2*h_third, w_third:2*w_third],  # Center
                face_roi[h_third:2*h_third, 2*w_third:], # Mid-right
                face_roi[2*h_third:, 0:w_third],         # Bottom-left
                face_roi[2*h_third:, w_third:2*w_third], # Bottom-center
                face_roi[2*h_third:, 2*w_third:]         # Bottom-right
            ]
            
            region_means = []
            for region in regions:
                if region.size > 0:
                    region_means.append(np.mean(region))
            
            if len(region_means) < 5:
                return False
            
            # Check for natural lighting variation
            lighting_std = np.std(region_means)
            
            # Real faces have natural lighting gradients, flat photos are more uniform
            if lighting_std < 8:  # Too uniform lighting
                return True
                
            # Check for unnatural symmetry (photos often have perfect symmetry)
            left_half = face_roi[:, :w//2]
            right_half = cv2.flip(face_roi[:, w//2:], 1)  # Flip horizontally
            
            if left_half.shape == right_half.shape:
                # Resize to match if needed
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
                
                # Calculate similarity
                diff = cv2.absdiff(left_half, right_half)
                symmetry_score = 1.0 - (np.mean(diff) / 255.0)
                
                if symmetry_score > 0.92:  # Too symmetric
                    return True
            
            return False
            
        except Exception as e:
            print(f"[WARN] 3D face structure analysis error: {e}")
            return False

# Global liveness detector instance
liveness_detector = LivenessDetector()

def enhanced_liveness_detection(image_bytes):
    """
    FIXED Enhanced liveness detection - returns True if suspicious (spoof), False if likely live
    Much more aggressive at detecting photos and printed images
    """
    try:
        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return True  # Suspicious if can't decode
        
        # Basic image quality check
        if img.shape[0] < 60 or img.shape[1] < 60:
            return True  # Too small to be reliable
        
        # Get face region
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = []
        
        if liveness_detector.face_cascade is not None:
            faces = liveness_detector.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
            )
        
        # If cascade detection fails, try to extract center region
        if len(faces) == 0:
            h, w = img.shape[:2]
            # Assume face is in center 70% of image
            x, y = int(w * 0.15), int(h * 0.15)
            w_face, h_face = int(w * 0.7), int(h * 0.7)
            face_region = img[y:y+h_face, x:x+w_face]
        else:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face_region = img[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return True  # No valid face region
        
        suspicious_score = 0
        
        # 1. FIXED Basic variance and sharpness check (more aggressive)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        std = float(np.std(gray))
        
        # Photos often have either very high sharpness OR very low variance
        if variance > 1500 or variance < 50:  # Too sharp OR too blurry
            suspicious_score += 3
        if std < 12 or std > 80:  # Too uniform OR too contrasty
            suspicious_score += 2
        
        # 2. FIXED Image quality and sharpness analysis
        if liveness_detector.check_image_quality_and_sharpness(img):
            suspicious_score += 4  # High weight - strong indicator
        
        # 3. FIXED Skin tone analysis on face region
        if liveness_detector.check_skin_tone_analysis(face_region):
            suspicious_score += 3
        
        # 4. FIXED Advanced texture pattern analysis
        if liveness_detector.check_texture_patterns(face_region):
            suspicious_score += 3
        
        # 5. FIXED 3D face structure analysis
        if liveness_detector.check_3d_face_structure(img):
            suspicious_score += 3
        
        # 6. FIXED Edge analysis - photos have artificial edges
        edges = cv2.Canny(gray, 30, 100)
        edge_density = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])
        
        if edge_density > 0.12 or edge_density < 0.015:  # Too many sharp edges OR too few
            suspicious_score += 2
        
        # 7. FIXED Color distribution analysis
        if len(img.shape) == 3:
            # Check for unnatural color distribution (printing artifacts)
            for channel in range(3):
                channel_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
                # Normalize histogram
                channel_hist = channel_hist.flatten() / np.sum(channel_hist)
                
                # Check for unnatural peaks
                peaks = np.where(channel_hist > 0.05)[0]  # Significant peaks
                if len(peaks) < 5:  # Too few color levels
                    suspicious_score += 1
        
        # 8. FIXED Frequency domain analysis
        try:
            f_transform = np.fft.fft2(gray)
            f_magnitude = np.abs(f_transform)
            
            # Check for artificial frequency patterns common in printed images
            low_freq_energy = np.sum(f_magnitude[:gray.shape[0]//8, :gray.shape[1]//8])
            total_energy = np.sum(f_magnitude)
            low_freq_ratio = low_freq_energy / (total_energy + 1e-6)
            
            if low_freq_ratio > 0.85:  # Too much low frequency content
                suspicious_score += 2
        except:
            pass
        
        # 9. Motion detection (if we have previous frames)
        liveness_detector.frame_buffer.append(img)
        has_motion = False
        if len(liveness_detector.frame_buffer) >= 2:
            has_motion = liveness_detector.detect_motion(
                liveness_detector.frame_buffer[-1], 
                liveness_detector.frame_buffer[-2]
            )
            if not has_motion and len(liveness_detector.frame_buffer) >= 3:
                suspicious_score += 4  # No motion is very suspicious
        
        # DECISION LOGIC: Much lower threshold for better photo detection
        # Threshold: 5 or more points = suspicious (was 3 before)
        is_suspicious = suspicious_score >= 5
        
        print(f"[DEBUG] Liveness check - Suspicious score: {suspicious_score}, Is suspicious: {is_suspicious}")
        return is_suspicious
        
    except Exception as e:
        print(f"[ERROR] Enhanced liveness detection: {e}")
        return True  # Default to suspicious on error

# Replace the basic_liveness_frame function
def basic_liveness_frame(data: bytes):
    """
    Updated function that uses enhanced liveness detection
    Returns True if suspicious (spoof/photo), False if likely live
    """
    return enhanced_liveness_detection(data)

# ------------------------
# Utility: password hashing
# ------------------------
def hash_password(password: str, salt: Optional[str] = None):
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150000)
    return salt, binascii.hexlify(dk).decode()

def verify_password(password: str, salt: str, hashed: str) -> bool:
    if not salt or not hashed:
        return False
    _, new_hash = hash_password(password, salt)
    return secrets.compare_digest(new_hash, hashed)

# ------------------------
# Database helpers
# ------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # students table now includes 'class' column
    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      avatar_url TEXT,
      seat_row INTEGER,
      seat_col INTEGER,
      mobile TEXT,
      class TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      student_id TEXT NOT NULL,
      type TEXT NOT NULL,
      ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      label TEXT,
      subject TEXT,
      room TEXT,
      note TEXT,
      FOREIGN KEY(student_id) REFERENCES students(id)
    )
    """)

    # NEW: Daily attendance records table for proper historical tracking
    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_attendance (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      student_id TEXT NOT NULL,
      attendance_date DATE NOT NULL,
      status TEXT NOT NULL DEFAULT 'absent',
      checkin_time TIMESTAMP,
      checkout_time TIMESTAMP,
      notes TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(student_id, attendance_date),
      FOREIGN KEY(student_id) REFERENCES students(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trust_scores (
      student_id TEXT PRIMARY KEY,
      score INTEGER DEFAULT 100,
      punctuality INTEGER DEFAULT 100,
      consistency INTEGER DEFAULT 100,
      streak INTEGER DEFAULT 0,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(student_id) REFERENCES students(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS insights (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      kind TEXT NOT NULL,
      text TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      impact TEXT DEFAULT 'low'
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      metric TEXT NOT NULL,
      week_start DATE NOT NULL,
      rows_json TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
      username TEXT PRIMARY KEY,
      password_hash TEXT NOT NULL,
      salt TEXT NOT NULL,
      role TEXT NOT NULL,
      display_name TEXT,
      student_id TEXT,
      assigned_classes TEXT
    )
    """)

    conn.commit()

    # seed demo data if needed
    cur.execute("SELECT COUNT(*) FROM students")
    if cur.fetchone()[0] == 0:
        # two students
        students = [
            ("sai", "Sai", "/avatars/sai.jpg", 1, 1, "92460118732", "A"),
            ("image_person", "Image Person", "/avatars/image_person.jpg", 1, 2, None, "A")
        ]
        for sid, name, avatar, r, c, mobile, cls in students:
            cur.execute("""
            INSERT OR REPLACE INTO students (id, name, avatar_url, seat_row, seat_col, mobile, class)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sid, name, avatar, r, c, mobile, cls))
            cur.execute("""
            INSERT OR REPLACE INTO trust_scores (student_id, score, punctuality, consistency, streak)
            VALUES (?, 100, 100, 100, 0)
            """, (sid,))

        insights = [
            ("trend", "📈 FIXED Enhanced liveness detection system initialized!", "high"),
            ("highlight", "🌟 Advanced photo detection now ACTIVE", "high"),
            ("prediction", "📊 Multi-factor anti-spoofing protection enabled", "med"),
            ("anomaly", "🔍 Aggressive photo and spoof detection ready", "med")
        ]
        cur.executemany("INSERT INTO insights (kind, text, impact) VALUES (?, ?, ?)", insights)

    # seed demo users
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        # HOD
        salt, hsh = hash_password("hodpass")
        cur.execute("INSERT INTO users (username, password_hash, salt, role, display_name) VALUES (?, ?, ?, ?, ?)",
                    ("hod", hsh, salt, "hod", "Head of Department"))
        # Teacher
        salt, hsh = hash_password("teacher1pass")
        cur.execute("INSERT INTO users (username, password_hash, salt, role, display_name, assigned_classes) VALUES (?, ?, ?, ?, ?, ?)",
                    ("teacher1", hsh, salt, "teacher", "Mrs. Teacher", "A,B"))
        # Parent (username = student's id 'sai', password = mobile)
        salt, hsh = hash_password("92460118732")
        cur.execute("INSERT INTO users (username, password_hash, salt, role, display_name, student_id) VALUES (?, ?, ?, ?, ?, ?)",
                    ("sai", hsh, salt, "parent", "Sai's Parent", "sai"))

    conn.commit()
    conn.close()
    print("[INFO] DB initialized / seeded with FIXED enhanced liveness detection")

# ------------------------
# NEW: Helper functions for daily attendance management
# ------------------------
def ensure_daily_attendance_record(student_id: str, attendance_date: date = None):
    """Ensure a daily attendance record exists for the student on the given date"""
    if attendance_date is None:
        attendance_date = date.today()
    
    conn = get_conn()
    cur = conn.cursor()
    
    # Check if record exists
    cur.execute("SELECT id FROM daily_attendance WHERE student_id = ? AND attendance_date = ?", 
                (student_id, attendance_date))
    if not cur.fetchone():
        # Create absent record by default
        cur.execute("""
            INSERT INTO daily_attendance (student_id, attendance_date, status) 
            VALUES (?, ?, 'absent')
        """, (student_id, attendance_date))
        conn.commit()
    
    conn.close()

def mark_student_present(student_id: str, attendance_date: date = None):
    """Mark a student as present for the given date"""
    if attendance_date is None:
        attendance_date = date.today()
    
    ensure_daily_attendance_record(student_id, attendance_date)
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE daily_attendance 
        SET status = 'present', 
            checkin_time = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE student_id = ? AND attendance_date = ?
    """, (student_id, attendance_date))
    conn.commit()
    conn.close()

def mark_student_absent(student_id: str, attendance_date: date = None):
    """Mark a student as absent for the given date"""
    if attendance_date is None:
        attendance_date = date.today()
    
    ensure_daily_attendance_record(student_id, attendance_date)
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE daily_attendance 
        SET status = 'absent',
            checkin_time = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE student_id = ? AND attendance_date = ?
    """, (student_id, attendance_date))
    conn.commit()
    conn.close()

def get_student_attendance_status(student_id: str, attendance_date: date = None):
    """Get the attendance status for a student on a specific date"""
    if attendance_date is None:
        attendance_date = date.today()
    
    ensure_daily_attendance_record(student_id, attendance_date)
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT status, checkin_time FROM daily_attendance 
        WHERE student_id = ? AND attendance_date = ?
    """, (student_id, attendance_date))
    row = cur.fetchone()
    conn.close()
    
    if row:
        return row["status"], row["checkin_time"]
    return "absent", None

# ------------------------
# Token helpers
# ------------------------
def create_token(username: str) -> str:
    token = uuid.uuid4().hex
    TOKENS[token] = username
    return token

def get_username_for_token(token: str) -> Optional[str]:
    return TOKENS.get(token)

def remove_token(token: str):
    TOKENS.pop(token, None)

# ------------------------
# Auth dependency
# ------------------------
def require_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization")
    token = authorization.split(" ", 1)[1]
    username = get_username_for_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, role, display_name, student_id, assigned_classes FROM users WHERE...  # previous code up to the truncated line")
    cur.execute("SELECT username, role, display_name, student_id, assigned_classes FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return {
        "username": row["username"],
        "role": row["role"],
        "display_name": row["display_name"],
        "student_id": row["student_id"],
        "assigned_classes": row["assigned_classes"]
    }

# ------------------------
# Known faces loader
# ------------------------
def load_known_faces():
    """
    Load face encodings from KNOWN_FACES_DIR and from students with avatar_url if available.
    This populates known_encodings and known_names global lists.
    """
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    # 1) Load images in KNOWN_FACES_DIR
    for file in os.listdir(KNOWN_FACES_DIR):
        p = KNOWN_FACES_DIR / file
        if not p.is_file():
            continue
        try:
            img = face_recognition.load_image_file(str(p))
            encs = face_recognition.face_encodings(img)
            if len(encs) > 0:
                known_encodings.append(encs[0])
                known_names.append(p.stem)
        except Exception as e:
            print(f"[WARN] Failed to load known face {p}: {e}")

    # 2) Load avatars referenced in DB students table (if not already included)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, avatar_url FROM students WHERE avatar_url IS NOT NULL")
        for sid, avatar_url in cur.fetchall():
            # avatar_url may be '/avatars/xxx.jpg' or similar
            if not avatar_url:
                continue
            filename = os.path.basename(avatar_url)
            path = AVATARS_DIR / filename
            if path.exists():
                try:
                    img = face_recognition.load_image_file(str(path))
                    encs = face_recognition.face_encodings(img)
                    if len(encs) > 0 and sid not in known_names:
                        known_encodings.append(encs[0])
                        known_names.append(sid)
                except Exception as e:
                    print(f"[WARN] Could not load avatar for {sid}: {e}")
        conn.close()
    except Exception as e:
        print(f"[WARN] Error loading avatars from DB: {e}")

    print(f"[INFO] Loaded {len(known_encodings)} known face(s).")


# ------------------------
# Auth endpoints
# ------------------------
@app.post("/api/login")
def api_login(username: str = Form(...), password: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    salt = row["salt"]
    stored_hash = row["password_hash"]
    if verify_password(password, salt, stored_hash):
        token = create_token(username)
        return {"token": token, "username": username}
    return JSONResponse({"error": "Invalid credentials"}, status_code=401)

@app.post("/api/logout")
def api_logout(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization")
    token = authorization.split(" ", 1)[1]
    remove_token(token)
    return {"ok": True}


# ------------------------
# Student management
# ------------------------
@app.get("/api/students", dependencies=[Depends(require_token)])
def api_get_students():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, avatar_url, seat_row, seat_col, mobile, class, created_at FROM students")
    rows = cur.fetchall()
    conn.close()
    students = [dict(r) for r in rows]
    return {"students": students}

@app.post("/api/students", dependencies=[Depends(require_token)])
async def api_add_student(
    id: str = Form(...),
    name: str = Form(...),
    seat_row: Optional[int] = Form(None),
    seat_col: Optional[int] = Form(None),
    mobile: Optional[str] = Form(None),
    class_name: Optional[str] = Form(None),
    avatar: Optional[UploadFile] = None
):
    avatar_url = None
    if avatar:
        ext = os.path.splitext(avatar.filename)[1] or ".jpg"
        safe_name = f"{id}{ext}"
        save_path = AVATARS_DIR / safe_name
        with open(save_path, "wb") as f:
            f.write(await avatar.read())
        avatar_url = f"/avatars/{safe_name}"

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO students (id, name, avatar_url, seat_row, seat_col, mobile, class)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (id, name, avatar_url, seat_row, seat_col, mobile, class_name))
    # ensure trust score exists
    cur.execute("""
        INSERT OR REPLACE INTO trust_scores (student_id, score, punctuality, consistency, streak)
        VALUES (?, COALESCE((SELECT score FROM trust_scores WHERE student_id = ?), 100), 
                COALESCE((SELECT punctuality FROM trust_scores WHERE student_id = ?), 100),
                COALESCE((SELECT consistency FROM trust_scores WHERE student_id = ?), 100),
                COALESCE((SELECT streak FROM trust_scores WHERE student_id = ?), 0)
        )
    """, (id, id, id, id))
    conn.commit()
    conn.close()

    # reload known faces to include new avatar
    load_known_faces()

    return {"ok": True, "id": id}


# ------------------------
# Recognition & Attendance endpoint (liveness protected)
# ------------------------
@app.post("/api/recognize")
async def api_recognize(image: UploadFile = File(...), room: Optional[str] = Form(None), subject: Optional[str] = Form(None), token_info: dict = Depends(require_token)):
    """
    Accepts an uploaded image, runs enhanced liveness detection, attempts to recognize faces,
    and marks attendance for matched students.
    """
    try:
        raw = await image.read()
        # Liveness check: returns True = suspicious/spoof
        is_spoof = basic_liveness_frame(raw)
        if is_spoof:
            return JSONResponse({"error": "Liveness check failed (suspected spoof/photo)."}, status_code=400)

        # Decode image for face_recognition processing
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        results = []
        conn = get_conn()
        cur = conn.cursor()

        for enc, loc in zip(face_encodings, face_locations):
            name = "Unknown"
            best_match_index = None
            if len(known_encodings) > 0:
                distances = face_recognition.face_distance(known_encodings, enc)
                best_idx = int(np.argmin(distances))
                best_distance = float(distances[best_idx])
                # flexible threshold: 0.45 strict, 0.6 fallback
                if best_distance < 0.45:
                    name = known_names[best_idx]
                elif best_distance < 0.6:
                    # double-check by checking DB id mapping (if name equals student id)
                    candidate = known_names[best_idx]
                    name = candidate  # still label it
                else:
                    name = "Unknown"

            # If matched to a student id, mark present & add event
            student_id = None
            if name != "Unknown":
                # name might be student id or file stem; attempt to find student in DB
                cur.execute("SELECT id FROM students WHERE id = ? OR name = ?", (name, name))
                r = cur.fetchone()
                if r:
                    student_id = r["id"]
                else:
                    # fallback: maybe name is file stem representing student id
                    student_id = name

            if student_id:
                # mark daily attendance present
                mark_student_present(student_id)
                # insert attendance_event
                cur.execute("""
                    INSERT INTO attendance_events (student_id, type, label, subject, room, note) 
                    VALUES (?, 'checkin', ?, ?, ?, ?)
                """, (student_id, "auto-checkin", subject, room, f"Recognized by {token_info['username']}"))
                conn.commit()

            results.append({
                "student_id": student_id,
                "label": name,
                "distance": None if len(known_encodings) == 0 else float(np.min(face_recognition.face_distance(known_encodings, enc))),
                "location": loc
            })

        conn.close()
        return {"ok": True, "results": results}
    except Exception as e:
        print(f"[ERROR] /api/recognize: {e}")
        return JSONResponse({"error": "Internal error during recognition"}, status_code=500)


# ------------------------
# Daily attendance APIs
# ------------------------
@app.get("/api/daily/{student_id}", dependencies=[Depends(require_token)])
def api_get_daily(student_id: str, attendance_date: Optional[str] = None):
    """
    Returns status and times for a student's daily attendance.
    attendance_date should be 'YYYY-MM-DD' if provided, defaults to today.
    """
    try:
        if attendance_date:
            dt = date.fromisoformat(attendance_date)
        else:
            dt = date.today()
    except Exception:
        return JSONResponse({"error": "Invalid date format. Use YYYY-MM-DD."}, status_code=400)

    status, checkin = get_student_attendance_status(student_id, dt)
    return {"student_id": student_id, "date": dt.isoformat(), "status": status, "checkin_time": checkin}


@app.post("/api/daily/{student_id}/mark_present", dependencies=[Depends(require_token)])
def api_mark_present(student_id: str):
    try:
        mark_student_present(student_id)
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, 'manual_checkin', 'marked_present')", (student_id,))
        conn.commit()
        conn.close()
        return {"ok": True, "student_id": student_id}
    except Exception as e:
        print(f"[ERROR] mark_present: {e}")
        return JSONResponse({"error": "Could not mark present"}, status_code=500)


# ------------------------
# Simple insights / leaderboard endpoints
# ------------------------
@app.get("/api/insights", dependencies=[Depends(require_token)])
def api_insights(limit: int = 20):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, kind, text, impact, created_at FROM insights ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return {"insights": [dict(r) for r in rows]}


@app.get("/api/leaderboard", dependencies=[Depends(require_token)])
def api_leaderboard(limit: int = 10):
    """
    Simple leaderboard by count of 'present' records over the last 7 days.
    """
    conn = get_conn()
    cur = conn.cursor()
    week_ago = (date.today() - timedelta(days=7)).isoformat()
    cur.execute("""
        SELECT s.id, s.name, COUNT(d.id) as presents
        FROM students s
        LEFT JOIN daily_attendance d ON d.student_id = s.id AND d.attendance_date >= ?
            AND d.status = 'present'
        GROUP BY s.id, s.name
        ORDER BY presents DESC
        LIMIT ?
    """, (week_ago, limit))
    rows = cur.fetchall()
    conn.close()
    return {"rows": [dict(r) for r in rows]}


# ------------------------
# Startup - initialize DB & load faces
# ------------------------
@app.on_event("startup")
def startup_event():
    init_db()
    load_known_faces()
    print("[INFO] PresenceAI started (FIXED enhanced liveness)")

# ------------------------
# Serve avatars & static fallback
# ------------------------
@app.get("/avatars/{filename}")
def serve_avatar(filename: str):
    path = AVATARS_DIR / filename
    if path.exists():
        return FileResponse(str(path))
    # fallback to default placeholder in static folder if present
    fallback = BASE_DIR / "static" / "avatar_placeholder.png"
    if fallback.exists():
        return FileResponse(str(fallback))
    raise HTTPException(status_code=404, detail="Avatar not found")

# ------------------------
# Run app
# ------------------------
if __name__ == "_main_":
    # Helpful defaults for local dev
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)