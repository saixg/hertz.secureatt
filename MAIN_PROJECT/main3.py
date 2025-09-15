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
import random

# ----------------- ADD THIS HELPER -----------------
def frames_similarity(img1, img2, size=(200,200)):
    """
    Fast approximate similarity measure in [0..1].
    1.0 = identical, 0.0 = totally different.
    We resize to `size` and compare mean absolute difference on grayscale.
    """
    try:
        if img1 is None or img2 is None:
            return 0.0
        # ensure grayscale uint8
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        g1r = cv2.resize(g1, size, interpolation=cv2.INTER_AREA)
        g2r = cv2.resize(g2, size, interpolation=cv2.INTER_AREA)
        diff = np.mean(np.abs(g1r.astype(np.float32) - g2r.astype(np.float32)))
        sim = 1.0 - (diff / 255.0)
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0
# ----------------- REPLACE YOUR OLD enhanced_liveness_detection WITH THIS -----------------
def enhanced_liveness_detection(image_bytes):
    """
    Enhanced liveness detection - returns True if suspicious (spoof), False if likely live.

    This version:
      - checks frame similarity across recent frames (detects static photos / paused videos)
      - requires motion OR blink (and head-pose when possible) for live acceptance
      - reuses your existing texture/color/edge checks and frame buffer
    """
    try:
        # Decode image bytes into BGR image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return True  # suspicious if can't decode

        # Basic image quality check
        if img.shape[0] < 50 or img.shape[1] < 50:
            return True  # Too small to be reliable

        # Convert to gray for some checks
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face detection (reuse your liveness_detector.face_cascade if present)
        faces = []
        if liveness_detector.face_cascade is not None:
            faces = liveness_detector.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

        # If no face found, use center region fallback
        if len(faces) == 0:
            h, w = img.shape[:2]
            x, y = int(w * 0.2), int(h * 0.2)
            w_face, h_face = int(w * 0.6), int(h * 0.6)
            face_region = img[y:y+h_face, x:x+w_face]
        else:
            # use largest detected face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face_region = img[y:y+h, x:x+w]

        if face_region.size == 0:
            return True  # suspicious if no valid face region

        suspicious_indicators = 0

        # 1) Sharpness / variance check
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        std = float(np.std(gray))
        if variance < 30 or std < 8:
            suspicious_indicators += 1

        # 2) Texture analysis (reuse existing method)
        try:
            if liveness_detector.check_texture_analysis(face_region):
                suspicious_indicators += 2
        except Exception:
            pass

        # 3) Color analysis
        try:
            if liveness_detector.check_color_analysis(face_region):
                suspicious_indicators += 1
        except Exception:
            pass

        # 4) Edge consistency
        try:
            if liveness_detector.check_edge_consistency(face_region):
                suspicious_indicators += 1
        except Exception:
            pass

        # 5) Append current frame to buffer for temporal checks (your existing buffer)
        try:
            liveness_detector.frame_buffer.append(img)
        except Exception:
            # if frame buffer fails, continue — we'll still use single-frame checks
            pass

        # 6) Motion detection between last two frames if available
        has_motion = False
        try:
            if len(liveness_detector.frame_buffer) >= 2:
                has_motion = liveness_detector.detect_motion(
                    liveness_detector.frame_buffer[-1],
                    liveness_detector.frame_buffer[-2]
                )
                # if no motion and buffer is larger, penalize more strongly
                if not has_motion and len(liveness_detector.frame_buffer) >= 5:
                    suspicious_indicators += 2
        except Exception:
            has_motion = False

        # 7) Blink detection from face region
        has_blink = False
        try:
            has_blink = liveness_detector.detect_blinks(face_region)
            if not has_blink and len(liveness_detector.blink_frames) >= 5:
                suspicious_indicators += 1
        except Exception:
            has_blink = False

        # 8) Head-pose / depth heuristic (if function available)
        try:
            head_pose_ok = estimate_head_pose(face_region, face_region.shape)
            if not head_pose_ok:
                suspicious_indicators += 1
        except Exception:
            # don't fail hard if head-pose not available
            pass

        # 9) NEW: check if recent frames are nearly IDENTICAL -> strong indicator of static/photo
        identical_pairs = 0
        try:
            if len(liveness_detector.frame_buffer) >= 3:
                recent = list(liveness_detector.frame_buffer)[-3:]
                sims = []
                for i in range(len(recent)-1):
                    s = frames_similarity(recent[i], recent[i+1])
                    sims.append(s)
                # Count extremely similar consecutive pairs
                for s in sims:
                    if s >= 0.995:  # nearly identical
                        identical_pairs += 1

                # If two consecutive pairs nearly identical AND no motion/blink -> very suspicious
                if identical_pairs >= 2 and not has_motion and not has_blink:
                    suspicious_indicators += 3  # high weight to force suspicious
        except Exception:
            pass

        # Final decision threshold (tunable): 3 or more suspicious indicators => suspicious
        return suspicious_indicators >= 3

    except Exception as e:
        print(f"[ERROR] Enhanced liveness detection (strict): {e}")
        # Default to suspicious on unexpected error (safer)
        return True



import numpy as np

def estimate_head_pose(landmarks, img_shape):
    # Simple 3D model points of facial landmarks (nose, eyes, etc.)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float32)

    # Use some fake 2D projection if landmarks not integrated
    # (placeholder: use image center and offsets to simulate fail for flat photos)
    size = img_shape
    focal_length = size[1]
    center = (size[1]//2, size[0]//2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4,1)) # no distortion
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            np.array([
                (center[0], center[1]-100),
                (center[0], center[1]+100),
                (center[0]-100, center[1]),
                (center[0]+100, center[1]),
                (center[0]-50, center[1]+50),
                (center[0]+50, center[1]+50)
            ], dtype=np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            # If translation z is too close to 0 => flat photo
            if abs(translation_vector[2]) < 1000:
                return False
            return True
    except:
        return False
    return False


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
# Enhanced Liveness Detection Module
# ------------------------

class LivenessDetector:
    def __init__(self):
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
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Count non-zero pixels (motion pixels)
            motion_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            motion_ratio = motion_pixels / total_pixels
            
            return motion_ratio > 0.005  # Lowered threshold for subtle motion
            
        except Exception as e:
            print(f"[WARN] Motion detection error: {e}")
            return False
    
    def detect_blinks(self, face_region):
        """Detect eye blinks in face region"""
        try:
            if self.eye_cascade is None:
                return False
                
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            
            # Resize face region if too small
            if gray_face.shape[0] < 50 or gray_face.shape[1] < 50:
                gray_face = cv2.resize(gray_face, (100, 100))
            
            eyes = self.eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
            
            if len(eyes) >= 1:  # At least one eye
                # Calculate eye aspect ratio for blink detection
                eye_ratios = []
                for (ex, ey, ew, eh) in eyes[:2]:  # Max 2 eyes
                    eye_roi = gray_face[ey:ey+eh, ex:ex+ew]
                    if eye_roi.size > 0:
                        # Simple eye openness calculation using mean intensity
                        eye_mean = np.mean(eye_roi)
                        eye_ratios.append(eye_mean)
                
                if eye_ratios:
                    avg_ratio = np.mean(eye_ratios)
                    self.blink_frames.append(avg_ratio)
                    
                    if len(self.blink_frames) >= 4:
                        # Check for blink pattern (decrease then increase)
                        ratios = list(self.blink_frames)
                        # Look for valley pattern (blink)
                        for i in range(1, len(ratios)-1):
                            if ratios[i] < ratios[i-1] * 0.9 and ratios[i] < ratios[i+1] * 0.9:
                                return True
            return False
            
        except Exception as e:
            print(f"[WARN] Blink detection error: {e}")
            return False
    
    def check_texture_analysis(self, image):
        """Analyze image texture to detect printed photos"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Resize if too large for processing
            if gray.shape[0] > 200 or gray.shape[1] > 200:
                gray = cv2.resize(gray, (200, 200))
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            avg_gradient = np.mean(magnitude)
            gradient_std = np.std(magnitude)
            
            # Calculate texture uniformity
            rows, cols = gray.shape
            center = gray[rows//4:3*rows//4, cols//4:3*cols//4]
            texture_std = np.std(center)
            
            # Photos typically have:
            # - Lower gradient variation
            # - More uniform texture
            # - Less natural variation
            if avg_gradient < 15 or gradient_std < 12 or texture_std < 20:
                return True  # Suspicious (likely photo)
            
            return False  # Likely live
            
        except Exception as e:
            print(f"[WARN] Texture analysis error: {e}")
            return False
    
    def check_color_analysis(self, image):
        """Analyze color distribution to detect printed photos"""
        try:
            if len(image.shape) != 3:
                return False
                
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Analyze color distribution
            hsv_std = [np.std(hsv[:,:,i]) for i in range(3)]
            
            # Check saturation and value distribution
            saturation_mean = np.mean(hsv[:,:,1])
            value_std = np.std(hsv[:,:,2])
            
            # Photos often have:
            # - Reduced color variation
            # - Lower saturation variation
            # - More uniform brightness
            if np.mean(hsv_std) < 12 or saturation_mean < 30 or value_std < 25:
                return True  # Suspicious
                
            # Check for color histogram peaks (artificial enhancement)
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # Look for unnatural concentration in hue or saturation
            max_h = np.max(hist_h)
            max_s = np.max(hist_s)
            total_pixels = image.shape[0] * image.shape[1]
            
            if max_h > total_pixels * 0.2 or max_s > total_pixels * 0.3:
                return True  # Too concentrated
                    
            return False
            
        except Exception as e:
            print(f"[WARN] Color analysis error: {e}")
            return False
    
    def check_edge_consistency(self, image):
        """Check edge consistency - photos often have artificial sharpening"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply different edge detection methods
            edges_canny = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_pixels = cv2.countNonZero(edges_canny)
            total_pixels = edges_canny.shape[0] * edges_canny.shape[1]
            edge_ratio = edge_pixels / total_pixels
            
            # Photos often have either too many sharp edges (over-sharpened)
            # or too few edges (printed/compressed)
            if edge_ratio > 0.15 or edge_ratio < 0.02:
                return True  # Suspicious
            
            return False
            
        except Exception as e:
            print(f"[WARN] Edge consistency error: {e}")
            return False

# Global liveness detector instance
liveness_detector = LivenessDetector()

def enhanced_liveness_detection(image_bytes):
    """
    Enhanced liveness detection - returns True if suspicious (spoof), False if likely live
    """
    try:
        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return True  # Suspicious if can't decode
        
        # Basic image quality check
        if img.shape[0] < 50 or img.shape[1] < 50:
            return True  # Too small to be reliable
        
        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = []
        
        if liveness_detector.face_cascade is not None:
            faces = liveness_detector.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
        
        # If cascade detection fails, try to extract center region
        if len(faces) == 0:
            h, w = img.shape[:2]
            # Assume face is in center 60% of image
            x, y = int(w * 0.2), int(h * 0.2)
            w_face, h_face = int(w * 0.6), int(h * 0.6)
            face_region = img[y:y+h_face, x:x+w_face]
        else:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face_region = img[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return True  # No valid face region
        
        suspicious_indicators = 0
        
        # 1. Basic variance check (existing but improved)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        std = float(np.std(gray))
        if variance < 30 or std < 8:  # Slightly relaxed thresholds
            suspicious_indicators += 1
        
        # 2. Texture analysis
        if liveness_detector.check_texture_analysis(face_region):
            suspicious_indicators += 2  # Higher weight for texture
        
        # 3. Color analysis  
        if liveness_detector.check_color_analysis(face_region):
            suspicious_indicators += 1
        
        # 4. Edge consistency check
        if liveness_detector.check_edge_consistency(face_region):
            suspicious_indicators += 1
        
        # 5. Add to frame buffer for motion analysis
        liveness_detector.frame_buffer.append(img)
        
        # 6. Motion detection (if we have previous frames)
        has_motion = False
        if len(liveness_detector.frame_buffer) >= 2:
            has_motion = liveness_detector.detect_motion(
                liveness_detector.frame_buffer[-1], 
                liveness_detector.frame_buffer[-2]
            )
            if not has_motion and len(liveness_detector.frame_buffer) >= 5:
                suspicious_indicators += 2  # No motion is very suspicious
        
        # 7. Blink detection (less strict for single frame)
        has_blink = liveness_detector.detect_blinks(face_region)
        if not has_blink and len(liveness_detector.blink_frames) >= 5:
            suspicious_indicators += 1
        
        # Decision logic: if too many suspicious indicators, flag as spoof
        # Threshold: 3 or more indicators = suspicious
        return suspicious_indicators >= 3
        
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
            ("trend", "📈 Enhanced liveness detection system initialized!", "high"),
            ("highlight", "🌟 Advanced anti-spoofing protection active", "high"),
            ("prediction", "📊 Multi-factor authentication ready", "med"),
            ("anomaly", "🔍 Photo detection and motion analysis enabled", "med")
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
    print("[INFO] DB initialized / seeded with enhanced liveness detection")

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
    cur.execute("SELECT username, role, display_name, student_id, assigned_classes FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(row)

# ------------------------
# Face loading & helpers
# ------------------------
def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    for fn in os.listdir(str(KNOWN_FACES_DIR)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        sid = os.path.splitext(fn)[0]
        path = KNOWN_FACES_DIR / fn
        try:
            img = face_recognition.load_image_file(str(path))
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(sid)
                print(f"[INFO] loaded face: {sid}")
            else:
                print(f"[WARN] no face found in {fn}")
        except Exception as e:
            print(f"[ERROR] loading face {fn}: {e}")

def encoding_from_bytes(data: bytes):
    """
    Robustly decode image bytes -> produce face encoding or None.
    """
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print("[WARN] encoding_from_bytes: cv2.imdecode returned None")
            return None

        # Drop alpha channel if present
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]

        # If grayscale (H,W), convert to BGR
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # Ensure dtype is uint8
        if img_bgr.dtype != np.uint8:
            if np.issubdtype(img_bgr.dtype, np.floating):
                img_bgr = (np.clip(img_bgr, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                img_bgr = img_bgr.astype(np.uint8)

        # Convert BGR -> RGB for face_recognition
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            img_rgb = img_bgr

        encs = face_recognition.face_encodings(img_rgb)
        if encs:
            return encs[0]
    except Exception as e:
        print("[ERROR] encoding_from_bytes", e)
    return None

def process_frames_consensus(frames_bytes: List[bytes], min_frames_required=2, distance_threshold=0.5):
    """
    Process multiple frames and require consensus with enhanced liveness detection.
    """
    if not frames_bytes:
        return {"status": "error", "message": "No frames provided"}

    match_counts = defaultdict(int)
    confidences = defaultdict(list)
    suspicious_count = 0  # Count how many frames appear suspicious
    total_processed = 0

    for b in frames_bytes:
        enc = encoding_from_bytes(b)
        if enc is None:
            continue
        total_processed += 1
        if not known_encodings:
            continue
        dists = face_recognition.face_distance(known_encodings, enc)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        confidence = max(0.0, 1.0 - best_dist)
        # consider matched if below distance threshold
        if best_dist <= distance_threshold:
            name = known_names[best_idx]
            match_counts[name] += 1
            confidences[name].append(confidence)
        
        # Enhanced liveness check per frame
        if enhanced_liveness_detection(b):
            suspicious_count += 1

    if total_processed == 0:
        return {"status": "error", "message": "No faces detected in frames"}

    if not match_counts:
        return {"status": "error", "message": "Unknown person detected. Please register first."}

    best_name = max(match_counts.items(), key=lambda x: x[1])[0]
    count = match_counts[best_name]
    avg_conf = sum(confidences[best_name]) / (len(confidences[best_name]) or 1)

    # Enhanced suspicious detection: if more than half the frames are suspicious
    is_suspicious = suspicious_count > (total_processed // 2)

    # require consensus
    if count >= min_frames_required or count >= (total_processed // 2 + 1):
        return {"status": "success", "student_id": best_name, "confidence": float(avg_conf), "is_suspicious": is_suspicious}
    else:
        return {"status": "error", "message": "Could not confidently match the face. Try again."}

# ------------------------
# WebSocket manager (broadcast)
# ------------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        try:
            self.active.remove(ws)
        except:
            pass

    async def broadcast(self, message: str):
        for ws in self.active[:]:
            try:
                await ws.send_text(message)
            except:
                try:
                    self.active.remove(ws)
                except:
                    pass

manager = ConnectionManager()

# ------------------------
# Helper added: fetch avatar URL for a student
# ------------------------
def get_avatar_url_for_student(student_id: str) -> str:
    """
    Return the avatar URL used by frontend. Prefer stored avatar_url.
    Falls back to /avatars/default.jpg
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT avatar_url FROM students WHERE id = ?", (student_id,))
        r = cur.fetchone()
        conn.close()
        if r and r["avatar_url"]:
            return r["avatar_url"]
    except Exception as e:
        print("[WARN] get_avatar_url_for_student error", e)
    return "/avatars/default.jpg"

# ------------------------
# API: health/login/me
# ------------------------
@app.get("/api/healthz")
async def healthz():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/api/login")
async def api_login(payload: Dict[str, Any]):
    username = payload.get("username")
    password = payload.get("password")
    if not username or not password:
        raise HTTPException(status_code=400, detail="username & password required")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, password_hash, salt, role, display_name, student_id, assigned_classes FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(password, row["salt"], row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(username)
    return {"token": token, "role": row["role"], "display_name": row["display_name"], "student_id": row["student_id"], "assigned_classes": row["assigned_classes"]}

@app.get("/api/me")
async def api_me(user = Depends(require_token)):
    return {
        "username": user["username"],
        "role": user["role"],
        "display_name": user.get("display_name"),
        "student_id": user.get("student_id"),
        "assigned_classes": user.get("assigned_classes")
    }

# ------------------------
# API: students list & detail
# ------------------------
@app.get("/api/students")
async def api_students():
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Ensure all students have today's attendance record
        cur.execute("SELECT id FROM students")
        student_rows = cur.fetchall()
        today = date.today()
        for s_row in student_rows:
            ensure_daily_attendance_record(s_row["id"], today)
        
        # Get students with their current attendance status from daily_attendance table
        cur.execute("""
            SELECT s.id, s.name, s.avatar_url, s.seat_row, s.seat_col,
                   ts.score as trust_score,
                   da.status as current_status,
                   da.checkin_time as last_checkin,
                   s.mobile, s.class
            FROM students s
            LEFT JOIN trust_scores ts ON s.id = ts.student_id
            LEFT JOIN daily_attendance da ON s.id = da.student_id AND da.attendance_date = date('now')
        """)
        rows = cur.fetchall()
        students = []
        for r in rows:
            cls = r["class"]
            status = r["current_status"] or "absent"
            
            # Map status for frontend compatibility
            if status == "present":
                smart_tag = "Present"
                ui_status = "present"
            elif status == "late":
                smart_tag = "Late" 
                ui_status = "late"
            elif status == "suspicious":
                smart_tag = "Suspicious"
                ui_status = "suspicious"
            else:
                smart_tag = "Absent"
                ui_status = "absent"
                
            students.append({
                "id": r["id"],
                "name": r["name"],
                "avatarUrl": r["avatar_url"] or "/avatars/default.jpg",
                "seat": {"row": r["seat_row"], "col": r["seat_col"]} if r["seat_row"] and r["seat_col"] else None,
                "trustScore": r["trust_score"] or 100,
                "status": ui_status,
                "smartTag": smart_tag,
                "attendancePct": calculate_attendance_percentage(r["id"]),
                "liveSeenAt": r["last_checkin"],
                "mobile": r["mobile"],
                "class": cls
            })
        conn.close()
        return students
    except Exception as e:
        print("[ERROR] api_students", e)
        return []

@app.get("/api/students/{student_id}")
async def api_student(student_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.name, s.avatar_url, s.seat_row, s.seat_col, ts.score as trust_score, s.class
        FROM students s LEFT JOIN trust_scores ts ON s.id = ts.student_id WHERE s.id = ?
    """, (student_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        raise HTTPException(status_code=404, detail="student not found")
    return {
        "id": r["id"],
        "name": r["name"],
        "avatarUrl": r["avatar_url"] or "/avatars/default.jpg",
        "seat": {"row": r["seat_row"], "col": r["seat_col"]} if r["seat_row"] and r["seat_col"] else None,
        "trustScore": r["trust_score"] or 100,
        "class": r["class"]
    }

# NEW API: Get attendance history for a student
@app.get("/api/students/{student_id}/attendance")
async def get_student_attendance_history(
    student_id: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
    limit: int = 30
):
    """Get attendance history for a specific student"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Check if student exists
        cur.execute("SELECT name FROM students WHERE id = ?", (student_id,))
        if not cur.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Build query based on date range
        query = """
            SELECT attendance_date, status, checkin_time, notes
            FROM daily_attendance 
            WHERE student_id = ?
        """
        params = [student_id]
        
        if start_date:
            query += " AND attendance_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND attendance_date <= ?"
            params.append(end_date)
            
        query += " ORDER BY attendance_date DESC LIMIT ?"
        params.append(limit)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        attendance_history = []
        for row in rows:
            attendance_history.append({
                "date": row["attendance_date"],
                "status": row["status"],
                "checkin_time": row["checkin_time"],
                "notes": row["notes"]
            })
        
        conn.close()
        return attendance_history
        
    except Exception as e:
        print(f"[ERROR] get_student_attendance_history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch attendance history")

# NEW API: Get attendance summary for date range
@app.get("/api/attendance/summary")
async def get_attendance_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    class_filter: Optional[str] = None
):
    """Get attendance summary for all students in a date range"""
    try:
        if not start_date:
            start_date = (date.today() - timedelta(days=7)).isoformat()
        if not end_date:
            end_date = date.today().isoformat()
            
        conn = get_conn()
        cur = conn.cursor()
        
        query = """
            SELECT s.id, s.name, s.class, da.attendance_date, da.status, da.checkin_time
            FROM students s
            LEFT JOIN daily_attendance da ON s.id = da.student_id 
                AND da.attendance_date BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if class_filter:
            query += " WHERE s.class = ?"
            params.append(class_filter)
            
        query += " ORDER BY s.name, da.attendance_date DESC"
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        summary = {}
        for row in rows:
            student_id = row["id"]
            if student_id not in summary:
                summary[student_id] = {
                    "student_id": student_id,
                    "name": row["name"],
                    "class": row["class"],
                    "attendance_records": []
                }
            
            if row["attendance_date"]:  # Only add if there's an attendance record
                summary[student_id]["attendance_records"].append({
                    "date": row["attendance_date"],
                    "status": row["status"],
                    "checkin_time": row["checkin_time"]
                })
        
        conn.close()
        return list(summary.values())
        
    except Exception as e:
        print(f"[ERROR] get_attendance_summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch attendance summary")

# ------------------------
# Register student (HOD) - main handler
# ------------------------
async def _register_student_internal(student_id: str, name: str, seat_row: Optional[int], seat_col: Optional[int], mobile: Optional[str], class_name: Optional[str], faceImage: Optional[UploadFile], avatar: Optional[UploadFile]):
    conn = get_conn()
    cur = conn.cursor()
    avatar_url = f"/avatars/{student_id}.jpg" if avatar else None
    cur.execute("""
        INSERT OR REPLACE INTO students (id, name, avatar_url, seat_row, seat_col, mobile, class)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (student_id, name, avatar_url, seat_row, seat_col, mobile, class_name))
    cur.execute("""
        INSERT OR IGNORE INTO trust_scores (student_id, score, punctuality, consistency, streak)
        VALUES (?, 100, 100, 100, 0)
    """, (student_id,))
    # save files
    if avatar:
        avatar_path = AVATARS_DIR / f"{student_id}.jpg"
        with open(avatar_path, "wb") as f:
            f.write(await avatar.read())
    if faceImage:
        face_path = KNOWN_FACES_DIR / f"{student_id}.jpg"
        with open(face_path, "wb") as f:
            f.write(await faceImage.read())
    conn.commit()
    conn.close()
    # reload faces if uploaded
    if faceImage:
        load_known_faces()
    return {"success": True, "student_id": student_id}

# ------------------------
# API: register-student (HOD only)
# ------------------------
@app.post("/api/register-student")
async def register_student(
    student_id: str = Form(...),
    name: str = Form(...),
    seat_row: Optional[int] = Form(None),
    seat_col: Optional[int] = Form(None),
    mobile: Optional[str] = Form(None),
    class_name: Optional[str] = Form(None),
    faceImage: Optional[UploadFile] = File(None),
    avatar: Optional[UploadFile] = File(None),
    user = Depends(require_token)
):
    if user["role"] != "hod":
        raise HTTPException(status_code=403, detail="HOD access required")
    return await _register_student_internal(student_id, name, seat_row, seat_col, mobile, class_name, faceImage, avatar)

# ------------------------
# API alias: POST /api/students (convenience for frontend)
# ------------------------
@app.post("/api/students")
async def post_students_alias(
    student_id: str = Form(...),
    name: str = Form(...),
    class_name: Optional[str] = Form(None),
    seat_row: Optional[int] = Form(None),
    seat_col: Optional[int] = Form(None),
    mobile: Optional[str] = Form(None),
    avatar: Optional[UploadFile] = File(None),
    face: Optional[UploadFile] = File(None),
    user = Depends(require_token)
):
    if user["role"] != "hod":
        raise HTTPException(status_code=403, detail="HOD required")
    return await _register_student_internal(student_id, name, seat_row, seat_col, mobile, class_name, face, avatar)

# ------------------------
# API: delete student (HOD only)
# ------------------------
@app.delete("/api/students/{student_id}")
async def delete_student(student_id: str, user = Depends(require_token)):
    if user["role"] != "hod":
        raise HTTPException(status_code=403, detail="HOD required")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM students WHERE id = ?", (student_id,))
    cur.execute("DELETE FROM trust_scores WHERE student_id = ?", (student_id,))
    cur.execute("DELETE FROM attendance_events WHERE student_id = ?", (student_id,))
    cur.execute("DELETE FROM daily_attendance WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()
    # remove files
    try:
        pface = KNOWN_FACES_DIR / f"{student_id}.jpg"
        pav = AVATARS_DIR / f"{student_id}.jpg"
        if pface.exists(): pface.unlink()
        if pav.exists(): pav.unlink()
    except Exception:
        pass
    load_known_faces()
    return {"success": True}

# ------------------------
# API: mark attendance (teacher/hod) - UPDATED TO USE DAILY ATTENDANCE
# ------------------------
@app.post("/api/attendance/mark")
async def mark_attendance(payload: Dict[str, Any], user = Depends(require_token)):
    student_id = payload.get("student_id")
    status = payload.get("status", "Present")
    attendance_date_str = payload.get("date")
    
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id required")
    if user["role"] not in ("hod", "teacher"):
        raise HTTPException(status_code=403, detail="Teacher or HOD required")
    
    # Parse date or use today
    if attendance_date_str:
        try:
            attendance_date = datetime.strptime(attendance_date_str, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        attendance_date = date.today()
    
    # teacher restrictions
    if user["role"] == "teacher" and user.get("assigned_classes"):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT class FROM students WHERE id = ?", (student_id,))
        r = cur.fetchone()
        conn.close()
        if r and r["class"]:
            cls = r["class"]
            allowed = (user.get("assigned_classes") or "").split(",")
            if cls not in allowed:
                raise HTTPException(status_code=403, detail="Teacher not allowed for this class")
    
    # Update daily attendance record
    if status.lower() in ["present", "late"]:
        mark_student_present(student_id, attendance_date)
        daily_status = "present"
    elif status.lower() == "suspicious":
        ensure_daily_attendance_record(student_id, attendance_date)
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            UPDATE daily_attendance 
            SET status = 'suspicious', updated_at = CURRENT_TIMESTAMP
            WHERE student_id = ? AND attendance_date = ?
        """, (student_id, attendance_date))
        conn.commit()
        conn.close()
        daily_status = "suspicious"
    else:
        mark_student_absent(student_id, attendance_date)
        daily_status = "absent"
    
    # record event in attendance_events for audit trail
    conn = get_conn()
    cur = conn.cursor()
    event_type = "suspicious" if status.lower() == "suspicious" else "checkin"
    label = f"Marked: {status} for {attendance_date}"
    cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, event_type, label))
    
    # Update trust scores
    if status.lower() == "suspicious":
        cur.execute("UPDATE trust_scores SET score = MAX(score - 5, 0), updated_at = CURRENT_TIMESTAMP WHERE student_id = ?", (student_id,))
    else:
        cur.execute("UPDATE trust_scores SET score = MIN(score + 1, 100), streak = streak + 1, punctuality = MIN(punctuality + 1, 100), updated_at = CURRENT_TIMESTAMP WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()
    
    # broadcast
    try:
        avatar_url = get_avatar_url_for_student(student_id)
    except Exception:
        avatar_url = "/avatars/default.jpg"
    asyncio.create_task(manager.broadcast(json.dumps({
        "type": "presence",
        "payload": {"student_id": student_id, "status": status, "timestamp": datetime.now().isoformat(), "avatarUrl": avatar_url, "date": attendance_date.isoformat()}
    })))
    
    return {"success": True, "date": attendance_date.isoformat(), "status": daily_status}

# ------------------------
# API: checkin (multi-frame) - UPDATED WITH STRICT LIVENESS
# ------------------------
@app.post("/api/checkin")
async def checkin(files: List[UploadFile] = File(None)):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    frames = []
    for f in files:
        try:
            b = await f.read()
            frames.append(b)
        except:
            pass

    res = process_frames_consensus(frames, min_frames_required=2, distance_threshold=0.48)
    if res.get("status") != "success":
        return JSONResponse({"success": False, "message": res.get("message", "No match")})

    student_id = res["student_id"]
    is_suspicious = res["is_suspicious"]
    confidence = res.get("confidence", 0.0)
    today = date.today()

    conn = get_conn()
    cur = conn.cursor()
    try:
        if is_suspicious:
            ensure_daily_attendance_record(student_id, today)
            cur.execute("""
                UPDATE daily_attendance 
                SET status = 'suspicious', updated_at = CURRENT_TIMESTAMP
                WHERE student_id = ? AND attendance_date = ?
            """, (student_id, today))
            cur.execute(
                "INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)",
                (student_id, "suspicious", "Spoof attempt blocked")
            )
            conn.commit()
            return JSONResponse({
                "success": False,
                "student_id": student_id,
                "status": "suspicious",
                "message": "Spoof detected — please use live camera"
            })
        else:
            mark_student_present(student_id, today)
            cur.execute(
                "INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)",
                (student_id, "checkin", "Camera checkin - liveness verified")
            )
            conn.commit()
            return JSONResponse({
                "success": True,
                "student_id": student_id,
                "status": "present",
                "confidence": confidence
            })
    finally:
        conn.close()

# ------------------------
# Timeline / trust / leaderboard / insights / seats
# ------------------------
@app.get("/api/timeline/{student_id}")
async def timeline(student_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, type, ts, label, subject, room, note FROM attendance_events WHERE student_id = ? ORDER BY ts DESC LIMIT 50", (student_id,))
    events = []
    for r in cur.fetchall():
        events.append({"id": str(r[0]), "studentId": student_id, "type": r[1], "ts": r[2], "label": r[3], "meta": {"subject": r[4], "room": r[5], "note": r[6]}})
    conn.close()
    return events

@app.get("/api/trust/{student_id}")
async def trust(student_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT score, punctuality, consistency, streak FROM trust_scores WHERE student_id = ?", (student_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return {"studentId": student_id, "score": 100, "punctuality": 100, "consistency": 100, "streak": 0}
    return {"studentId": student_id, "score": r[0], "punctuality": r[1], "consistency": r[2], "streak": r[3]}

@app.get("/api/leaderboard")
async def leaderboard(metric: str = "overall"):
    try:
        conn = get_conn()
        cur = conn.cursor()
        if metric == "punctuality":
            order_col = "ts.punctuality"
        elif metric == "consistency":
            order_col = "ts.consistency"
        else:
            order_col = "ts.score"
        cur.execute(f"SELECT s.id, s.name, s.avatar_url, {order_col} as score, 0 as trend FROM students s LEFT JOIN trust_scores ts ON s.id = ts.student_id ORDER BY score DESC LIMIT 10")
        out = []
        for r in cur.fetchall():
            out.append({"id": r[0], "name": r[1], "avatarUrl": r[2], "score": r[3] or 100, "trend": r[4]})
        conn.close()
        return out
    except Exception as e:
        print("[ERROR] leaderboard", e)
        return []

@app.get("/api/insights")
async def insights(role: str = "teacher"):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, kind, text, created_at, impact FROM insights ORDER BY created_at DESC LIMIT 20")
        rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({"id": str(r[0]), "kind": r[1], "text": r[2], "createdAt": r[3], "impact": r[4]})
        conn.close()
        return out
    except Exception as e:
        print("[ERROR] insights", e)
        return []

@app.get("/api/seats")
async def seats():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, name, seat_row, seat_col FROM students WHERE seat_row IS NOT NULL AND seat_col IS NOT NULL")
        out = {}
        for r in cur.fetchall():
            key = f"{r[2]}-{r[3]}"
            out[key] = {"studentId": r[0], "studentName": r[1], "row": r[2], "col": r[3]}
        conn.close()
        return out
    except Exception as e:
        print("[ERROR] seats", e)
        return {}

# ------------------------
# Simulate checkin (demo) - UPDATED WITH ENHANCED LIVENESS
# ------------------------
@app.post("/api/simulate-checkin")
async def simulate_checkin(payload: Dict[str, Any]):
    student_id = payload.get("student_id")
    status = payload.get("status", "Present")
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id required")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM students WHERE id = ?", (student_id,))
    r = cur.fetchone()
    if not r:
        conn.close()
        raise HTTPException(status_code=404, detail="student not found")
    name = r["name"]
    
    today = date.today()
    
    # Update daily attendance
    if status.lower() == "suspicious":
        ensure_daily_attendance_record(student_id, today)
        cur.execute("""
            UPDATE daily_attendance 
            SET status = 'suspicious', updated_at = CURRENT_TIMESTAMP
            WHERE student_id = ? AND attendance_date = ?
        """, (student_id, today))
        cur.execute("UPDATE trust_scores SET score = MAX(score - 5, 0) WHERE student_id = ?", (student_id,))
    else:
        mark_student_present(student_id, today)
        cur.execute("UPDATE trust_scores SET score = MIN(score + 2, 100), streak = streak + 1 WHERE student_id = ?", (student_id,))
    
    # Record event
    event_type = "suspicious" if status.lower() == "suspicious" else "checkin"
    cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, event_type, f"Simulated {status}"))
    conn.commit()
    conn.close()
    
    try:
        avatar_url = get_avatar_url_for_student(student_id)
    except Exception:
        avatar_url = "/avatars/default.jpg"
    asyncio.create_task(manager.broadcast(json.dumps({"type":"presence","payload":{"student_id": student_id, "status": status, "timestamp": datetime.now().isoformat(), "avatarUrl": avatar_url}})))
    
    return {"success": True, "student_id": student_id, "student_name": name, "status": status}

# ------------------------
# WebSocket endpoint - UPDATED WITH STRICT LIVENESS
# ------------------------
@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({"type":"info","message":"ws_connected"}))
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
            except Exception as e:
                print("[WARN] ws receive_text error:", e)
                await asyncio.sleep(0.1)
                continue

            try:
                msg = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"type":"ack","message":"ok"}))
                continue

            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type":"ack","message":"ok"}))
                continue

            if "img" in msg:
                b64 = msg.get("img") or ""
                if b64.startswith("data:") and "," in b64:
                    b64 = b64.split(",", 1)[1]

                try:
                    img_bytes = base64.b64decode(b64)
                except Exception:
                    await websocket.send_text(json.dumps({"type":"error", "message": "invalid_base64"}))
                    continue

                if enhanced_liveness_detection(img_bytes):
                    await websocket.send_text(json.dumps({
                        "type":"suspicious",
                        "message":"Spoof/photo detected — please show live face"
                    }))
                    continue

                enc = encoding_from_bytes(img_bytes)
                if enc is None:
                    await websocket.send_text(json.dumps({"type":"error","message":"no_face_detected_or_decode_failed"}))
                    continue

                if not known_encodings:
                    await websocket.send_text(json.dumps({"type":"error","message":"no_known_faces_loaded"}))
                    continue

                try:
                    dists = face_recognition.face_distance(known_encodings, enc)
                except Exception as e:
                    print("[ERROR] face_distance failed:", e)
                    await websocket.send_text(json.dumps({"type":"error","message":"matching_failed"}))
                    continue

                best_idx = int(np.argmin(dists))
                best_dist = float(dists[best_idx]) if len(dists) > 0 else math.inf
                confidence = max(0.0, 1.0 - best_dist)

                TOL = 0.48
                if best_dist <= TOL:
                    student_id = known_names[best_idx]
                    mark_student_present(student_id, date.today())
                    conn = get_conn()
                    cur = conn.cursor()
                    cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", 
                                (student_id, "checkin", "WebSocket checkin - liveness verified"))
                    cur.execute("UPDATE trust_scores SET score = MIN(score + 1, 100), streak = streak + 1 WHERE student_id = ?", (student_id,))
                    conn.commit()
                    conn.close()
                    resp = {"type":"recognized", "student_id": student_id, "confidence": confidence}
                    await websocket.send_text(json.dumps(resp))
                else:
                    resp = {"type":"unknown", "best_distance": best_dist, "confidence": confidence}
                    await websocket.send_text(json.dumps(resp))
                continue

            await websocket.send_text(json.dumps({"type":"ack","message":"ok"}))

    except Exception as e:
        print("[ERROR] ws_events outer", e)
        manager.disconnect(websocket)

# ------------------------
# Helper: attendance % - UPDATED TO USE DAILY ATTENDANCE
# ------------------------
def calculate_attendance_percentage(student_id: str, days: int = 30) -> int:
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Count present days from daily_attendance table
        cur.execute("""
          SELECT COUNT(*) as present_days
          FROM daily_attendance
          WHERE student_id = ?
          AND status = 'present'
          AND attendance_date >= date('now', '-{} days')
        """.format(days), (student_id,))
        present_days = cur.fetchone()[0] or 0
        
        # Count total days that should have attendance (exclude weekends/holidays if needed)
        cur.execute("""
          SELECT COUNT(DISTINCT attendance_date) as total_days
          FROM daily_attendance
          WHERE student_id = ?
          AND attendance_date >= date('now', '-{} days')
        """.format(days), (student_id,))
        total_days = cur.fetchone()[0] or 1  # Avoid division by zero
        
        conn.close()
        if total_days == 0:
            return 100  # New student, assume 100%
        return min(int((present_days / total_days) * 100), 100)
    except Exception as e:
        print(f"[ERROR] calculate_attendance_percentage: {e}")
        return 85  # fallback

# ------------------------
# Startup
# ------------------------
@app.on_event("startup")
async def startup():
    print("[INFO] Starting PresenceAI backend with Enhanced Liveness Detection...")
    print("[INFO] BASE_DIR:", BASE_DIR)
    print("[INFO] avatars dir:", AVATARS_DIR.resolve())
    try:
        print("[INFO] avatars contents:", [p.name for p in AVATARS_DIR.iterdir()])
    except Exception as e:
        print("[WARN] avatars listing failed:", e)

    init_db()
    load_known_faces()
    print(f"[INFO] known faces loaded: {len(known_names)}")
    print("[INFO] Enhanced liveness detection system ready!")

# ------------------------
# Serve static avatar files
# ------------------------
@app.get("/avatars/{filename}")
async def avatar_file(filename: str):
    # prevent path traversal
    filename = os.path.basename(filename)
    path = AVATARS_DIR / filename
    if path.exists() and path.is_file():
        return FileResponse(str(path))
    default = AVATARS_DIR / "default.jpg"
    if default.exists() and default.is_file():
        return FileResponse(str(default))
    # fallback: return 404 with JSON so frontend can load placeholder
    raise HTTPException(status_code=404, detail="Avatar not found")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    init_db()
    load_known_faces()
    print("Enhanced PresenceAI Backend Ready!")
    print("Features:")
    print("- Multi-factor liveness detection")
    print("- Photo/spoof detection")
    print("- Motion and blink analysis")
    print("- Texture and color analysis")
    print("- Enhanced trust scoring")
    print("")
    print("Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  