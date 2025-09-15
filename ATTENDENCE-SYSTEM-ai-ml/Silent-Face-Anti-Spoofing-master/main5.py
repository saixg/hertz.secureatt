"""
PresenceAI - Attendance Backend (Face Recognition + Raw Anti-Spoof Ensemble)
- Uses your raw anti-spoof method: for each frame, loop over all .pth models,
  run model_test.predict(img, model_path) and sum predictions.
- If argmax(sum) != 1 -> SPOOF -> block immediately.
- Recognition and attendance only run when anti-spoof labels frame as REAL (label==1).
"""

import os
import sys
import cv2
import json
import base64
import uvicorn
import sqlite3
import pathlib
import binascii
import hashlib
import secrets
import uuid
import numpy as np
import face_recognition
import math
import asyncio
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------
# NOTES BEFORE RUNNING
# - Ensure Silent-Face-Anti-Spoofing-master is present and importable.
# - Ensure resources/anti_spoof_models contains your .pth files (MiniFASNetV2, MiniFASNetV1SE).
# - Ensure known_faces directory has images named by student_id (e.g. saigireesh.jpg).
# - Install dependencies: face_recognition, dlib, opencv-python, numpy, fastapi, uvicorn.
# ---------------------------

# Resolve base dir reliably
BASE_DIR = pathlib.Path(__file__).resolve().parent

# Directories
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
AVATARS_DIR = BASE_DIR / "avatars"
TEMP_DIR = BASE_DIR / "temp"
MODEL_DIR = BASE_DIR / "resources" / "anti_spoof_models"

for d in (KNOWN_FACES_DIR, AVATARS_DIR, TEMP_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# DB path
DB_PATH = str(BASE_DIR / "attendance.db")

# FastAPI init
app = FastAPI(title="PresenceAI Attendance Backend", version="2.0.0")
try:
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
except Exception:
    pass
try:
    app.mount("/avatars_static", StaticFiles(directory=str(AVATARS_DIR)), name="avatars_static")
except Exception:
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory tokens (demo)
TOKENS: Dict[str, str] = {}

# Known faces cache
known_encodings: List[np.ndarray] = []
known_names: List[str] = []

# Face matching tolerance
TOLERANCE = 0.45

# ---------------------------
# Anti-spoof (raw ML ensemble) - keep exactly your raw method
# ---------------------------
sys.path.append(str(BASE_DIR / "Silent-Face-Anti-Spoofing-master"))
try:
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name
except Exception as e:
    raise RuntimeError(f"Could not import Silent-Face-Anti-Spoofing modules: {e}")

# instantiate predictor and cropper
model_test = AntiSpoofPredict(0)
image_cropper = CropImage()

# load list of .pth models
anti_spoof_models = [m for m in os.listdir(str(MODEL_DIR)) if m.endswith(".pth")]
if not anti_spoof_models:
    raise RuntimeError("No anti-spoof models found in MODEL_DIR")
print(f"[INFO] Anti-spoof models: {anti_spoof_models}")

def is_frame_suspicious_by_ensemble(org_img_bgr, bbox):
    """
    Apply raw ensemble anti-spoof method exactly as your ML code:
    - For each .pth model, crop the bbox with CropImage and call model_test.predict(img, model_path).
    - Sum predictions into a vector and take argmax.
    - If argmax != 1 -> suspicious (block).
    Return True if suspicious (i.e., block), False if real.
    """
    left, top, right, bottom = bbox
    prediction = np.zeros((1, 3))
    for model_name in anti_spoof_models:
        model_path = os.path.join(str(MODEL_DIR), model_name)
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": org_img_bgr,
            "bbox": (left, top, right, bottom),
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        try:
            img = image_cropper.crop(**param)
            if img is None or img.size == 0:
                # skip model if cropping failed
                continue
            pred = model_test.predict(img, model_path)  # expected shape (1,3)
            # defensive: ensure pred shape
            pred = np.asarray(pred)
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)
            prediction += pred
        except Exception as e:
            # log and continue with next model
            print(f"[WARN] anti-spoof predict failed for {model_name}: {e}")
            continue

    if np.all(prediction == 0):
        # nothing produced by models -> treat as suspicious to be safe
        print("[WARN] No valid anti-spoof predictions - treating as suspicious")
        return True

    label = int(np.argmax(prediction))
    # label == 1 indicates real in your raw code
    is_suspicious = (label != 1)
    print(f"[DEBUG] Ensemble anti-spoof label={label} (suspicious={is_suspicious})")
    return is_suspicious

# ---------------------------
# DB helpers and init
# ---------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # keep original schema as in your old main.py (trimmed to essentials here)
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
    # seed demo data minimal (if empty)
    cur.execute("SELECT COUNT(*) FROM students")
    if cur.fetchone()[0] == 0:
        students = [
            ("yagensh", "Yagensh", "/avatars/yagensh.jpg", 1, 1, "1234567890", "A"),
            ("saigireesh", "Sai Gireesh", "/avatars/saigireesh.jpg", 1, 2, "2345678901", "A"),
            ("venkat", "Venkat", "/avatars/venkat.jpg", 1, 3, "3456789012", "A"),
            ("hasini", "Hasini", "/avatars/hasini.jpg", 1, 4, "4567890123", "A")
        ]
        for sid, name, avatar, r, c, mobile, cls in students:
            cur.execute("INSERT OR REPLACE INTO students (id, name, avatar_url, seat_row, seat_col, mobile, class) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (sid, name, avatar, r, c, mobile, cls))
            cur.execute("INSERT OR REPLACE INTO trust_scores (student_id, score, punctuality, consistency, streak) VALUES (?, 100, 100, 100, 0)",
                        (sid,))
    conn.commit()
    conn.close()
    print("[INFO] DB initialized")

# ---------------------------
# Password helpers
# ---------------------------
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

# ---------------------------
# Face loading and encoding utilities
# (kept from your original improved loader)
# ---------------------------
def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    if not os.path.exists(str(KNOWN_FACES_DIR)):
        print(f"[WARN] Known faces dir doesn't exist: {KNOWN_FACES_DIR}")
        return
    face_files = [f for f in os.listdir(str(KNOWN_FACES_DIR)) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))]
    if not face_files:
        print("[WARN] No face files found in known_faces")
        return
    for fn in face_files:
        sid = os.path.splitext(fn)[0]
        path = KNOWN_FACES_DIR / fn
        try:
            img = face_recognition.load_image_file(str(path))
            h,w = img.shape[:2]
            if h < 120 or w < 120:
                img = cv2.resize(img, (160,160))
            elif h>600 or w>600:
                scale = min(600/w, 600/h)
                img = cv2.resize(img, (int(w*scale), int(h*scale)))
            encodings = []
            # method 1 hog
            try:
                locs = face_recognition.face_locations(img, model='hog')
                if locs:
                    encs = face_recognition.face_encodings(img, locs, num_jitters=3)
                    if encs: encodings.extend(encs)
            except Exception as e:
                print(f"[DEBUG] HOG failed for {fn}: {e}")
            # method 2 cnn
            if not encodings:
                try:
                    locs = face_recognition.face_locations(img, model='cnn')
                    if locs:
                        encs = face_recognition.face_encodings(img, locs, num_jitters=3)
                        if encs: encodings.extend(encs)
                except Exception as e:
                    print(f"[DEBUG] CNN failed for {fn}: {e}")
            # method 3 enhanced
            if not encodings:
                try:
                    img_enh = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
                    locs = face_recognition.face_locations(img_enh, model='hog')
                    if locs:
                        encs = face_recognition.face_encodings(img_enh, locs, num_jitters=3)
                        if encs: encodings.extend(encs)
                except Exception as e:
                    print(f"[DEBUG] enhanced failed for {fn}: {e}")
            # method 4 assume whole image
            if not encodings:
                try:
                    encs = face_recognition.face_encodings(img, num_jitters=5)
                    if encs: encodings.extend(encs)
                except Exception as e:
                    print(f"[DEBUG] whole-image failed for {fn}: {e}")
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(sid)
                enc = encodings[0]
                print(f"[INFO] Loaded face {sid}: mean={enc.mean():.4f} std={enc.std():.4f}")
            else:
                print(f"[ERROR] Could not encode face file: {fn}")
        except Exception as e:
            print(f"[ERROR] Exception loading {fn}: {e}")
    print(f"[INFO] Finished loading faces. Count: {len(known_names)}")

def improved_encoding_from_bytes(data: bytes):
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h,w = img_rgb.shape[:2]
        if h<100 or w<100:
            sf = max(100/h, 100/w)
            img_rgb = cv2.resize(img_rgb, (int(w*sf), int(h*sf)))
        encodings = []
        try:
            locs = face_recognition.face_locations(img_rgb, model='hog')
            if locs:
                encs = face_recognition.face_encodings(img_rgb, locs)
                encodings.extend(encs)
        except Exception as e:
            print(f"[DEBUG] standard detection failed: {e}")
        if not encodings:
            try:
                img_enh = cv2.convertScaleAbs(img_rgb, alpha=1.1, beta=5)
                locs = face_recognition.face_locations(img_enh, model='hog')
                if locs:
                    encs = face_recognition.face_encodings(img_enh, locs)
                    encodings.extend(encs)
            except Exception as e:
                print(f"[DEBUG] enhanced detection failed: {e}")
        if not encodings:
            try:
                locs = face_recognition.face_locations(img_rgb, number_of_times_to_upsample=2)
                if locs:
                    encs = face_recognition.face_encodings(img_rgb, locs)
                    encodings.extend(encs)
            except Exception as e:
                print(f"[DEBUG] upsample failed: {e}")
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"[ERROR] improved_encoding_from_bytes: {e}")
        return None

# ---------------------------
# process_frames_consensus - uses anti-spoof ensemble per frame (raw method)
# ---------------------------
def process_frames_consensus(frames_bytes: List[bytes], min_frames_required=3, distance_threshold=0.5):
    if not frames_bytes:
        return {"status":"error","message":"No frames provided"}
    if len(frames_bytes) < min_frames_required:
        return {"status":"error","message":f"Need at least {min_frames_required} frames for verification"}
    if not known_encodings or not known_names:
        return {"status":"error","message":"No registered faces loaded"}

    match_counts = defaultdict(int)
    confidences = defaultdict(list)
    suspicious_count = 0
    total_processed = 0
    all_distances = defaultdict(list)

    # process each frame
    for i, fb in enumerate(frames_bytes):
        try:
            # decode frame bytes
            arr = np.frombuffer(fb, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[DEBUG] frame {i+1} decode failed")
                continue

            # minimal resize for performance
            # detect face locations (RGB)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(img_rgb)
            if not face_locs:
                print(f"[DEBUG] frame {i+1} no face found")
                continue

            # pick first face (assumption)
            top, right, bottom, left = face_locs[0]
            # scale coords to original image (face_recognition returned on full img since we passed full-size)
            bbox = (left, top, right, bottom)

            # anti-spoof check using raw ensemble
            try:
                is_susp = is_frame_suspicious_by_ensemble(img_bgr, bbox)
            except Exception as e:
                print(f"[WARN] ensemble check failed on frame {i+1}: {e}")
                is_susp = True

            if is_susp:
                suspicious_count += 1
                print(f"[DEBUG] frame {i+1} flagged suspicious - skipping recognition")
                continue

            # get improved encoding
            enc = improved_encoding_from_bytes(fb)
            if enc is None:
                print(f"[DEBUG] frame {i+1} encoding failed")
                continue

            total_processed += 1

            # face distance against known
            distances = face_recognition.face_distance(known_encodings, enc)
            for name, d in zip(known_names, distances):
                all_distances[name].append(d)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            best_name = known_names[best_idx]
            confidence = max(0.0, 1.0 - best_dist)

            if best_dist <= distance_threshold:
                match_counts[best_name] += 1
                confidences[best_name].append(confidence)
                print(f"[DEBUG] frame {i+1} matched {best_name} dist={best_dist:.4f}")
            else:
                print(f"[DEBUG] frame {i+1} no match best_dist={best_dist:.4f}")

        except Exception as e:
            print(f"[ERROR] processing frame {i+1}: {e}")
            continue

    # results analysis
    if total_processed == 0:
        return {"status":"error","message":"No valid frames after anti-spoof and detection"}

    suspicious_ratio = suspicious_count / len(frames_bytes)
    print(f"[DEBUG] suspicious_ratio={suspicious_ratio:.2%} processed={total_processed}")

    # if majority of frames suspicious -> block
    if suspicious_ratio > 0.7:
        return {"status":"error","message":"Multiple frames detected as photos. Please use live camera."}

    if not match_counts:
        # provide distances debug
        for name in known_names:
            if name in all_distances and all_distances[name]:
                print(f"[DEBUG]{name} distances: avg={sum(all_distances[name])/len(all_distances[name]):.4f} min={min(all_distances[name]):.4f}")
        return {"status":"error","message":"Person not recognized. Ensure you are registered."}

    best_name = max(match_counts.items(), key=lambda x: x[1])[0]
    match_count = match_counts[best_name]
    avg_conf = sum(confidences[best_name]) / len(confidences[best_name]) if confidences[best_name] else 0.0

    min_required = max(1, int(total_processed * 0.5))
    if match_count >= min_required:
        return {"status":"success","student_id":best_name,"confidence":float(avg_conf),"is_suspicious":False}
    else:
        return {"status":"error","message":"Insufficient consensus. Try again."}

# ---------------------------
# Simple WebSocket manager (as before)
# ---------------------------
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

# ---------------------------
# Helper: get_avatar_url_for_student
# ---------------------------
def get_avatar_url_for_student(student_id: str) -> str:
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT avatar_url FROM students WHERE id = ?", (student_id,))
        r = cur.fetchone()
        conn.close()
        if r and r["avatar_url"]:
            return r["avatar_url"]
    except Exception as e:
        print("[WARN] avatar fetch error:", e)
    return "/avatars/default.jpg"

def mark_student_present(student_id: str, attendance_date: date):
    """
    Mark the student present for the given date. This will insert a new
    daily_attendance row if missing, or update an existing row to 'present'
    while preserving an existing checkin_time (unless missing).
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        adate = attendance_date.isoformat() if isinstance(attendance_date, (date,datetime)) else str(attendance_date)
        now = datetime.now().isoformat()
        # Insert if not exists
        cur.execute(
            """INSERT OR IGNORE INTO daily_attendance
               (student_id, attendance_date, status, checkin_time, created_at, updated_at)
               VALUES (?, ?, 'present', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            (student_id, adate, now)
        )
        # Ensure existing row is marked present and has a checkin_time
        cur.execute(
            """UPDATE daily_attendance
               SET status = 'present',
                   checkin_time = COALESCE(checkin_time, ?),
                   updated_at = CURRENT_TIMESTAMP
               WHERE student_id = ? AND attendance_date = ?""",
            (now, student_id, adate)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("[ERROR] mark_student_present:", e)

# ---------------------------
# Auth dependency (simple token)
# ---------------------------
def create_token(username: str) -> str:
    token = uuid.uuid4().hex
    TOKENS[token] = username
    return token

def get_username_for_token(token: str) -> Optional[str]:
    return TOKENS.get(token)

def require_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization")
    token = authorization.split(" ",1)[1]
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

# ---------------------------
# Endpoints (health, login, me, students, verify faces, reload, test-face-recognition)
# Keep behavior and signatures from your original file.
# ---------------------------
@app.get("/api/healthz")
async def healthz():
    return {"status":"ok", "ts": datetime.now().isoformat()}

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
    return {"username": user["username"], "role": user["role"], "display_name": user.get("display_name"), "student_id": user.get("student_id"), "assigned_classes": user.get("assigned_classes")}

@app.get("/api/students")
async def api_students():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, name, avatar_url, seat_row, seat_col, mobile, class FROM students")
        rows = cur.fetchall()
        conn.close()
        students = []
        for r in rows:
            students.append({
                "id": r["id"],
                "name": r["name"],
                "avatarUrl": r["avatar_url"] or "/avatars/default.jpg",
                "seat": {"row": r["seat_row"], "col": r["seat_col"]} if r["seat_row"] else None,
                "mobile": r["mobile"],
                "class": r["class"]
            })
        return students
    except Exception as e:
        print("[ERROR] api_students:", e)
        return []

@app.get("/api/verify-faces")
async def verify_faces():
    face_files = []
    if os.path.exists(str(KNOWN_FACES_DIR)):
        face_files = [f for f in os.listdir(str(KNOWN_FACES_DIR)) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))]
    return {"known_faces_count": len(known_names), "known_faces": known_names, "files_in_dir": face_files}

@app.post("/api/reload-faces")
async def reload_faces(user = Depends(require_token)):
    if user["role"] not in ["hod","teacher"]:
        raise HTTPException(status_code=403, detail="HOD or Teacher access required")
    load_known_faces()
    return {"success": True, "loaded_faces_count": len(known_names), "loaded_faces": known_names}

@app.post("/api/test-face-recognition")
async def test_face_recognition(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if not known_encodings:
        return {"error":"No known faces loaded"}
    results = []
    for i,file in enumerate(files):
        try:
            content = await file.read()
            enc = improved_encoding_from_bytes(content)
            if enc is None:
                results.append({"file_index": i, "filename": file.filename, "status":"error","message":"No face detected"})
                continue
            distances = face_recognition.face_distance(known_encodings, enc)
            matches = []
            for name,dist in zip(known_names, distances):
                matches.append({"name": name, "distance": float(dist), "confidence": float(max(0,1-dist)), "match": dist<=0.5})
            matches.sort(key=lambda x: x["distance"])
            results.append({"file_index":i,"filename":file.filename,"status":"success","best_match":matches[0] if matches else None,"all_matches":matches})
        except Exception as e:
            results.append({"file_index":i,"filename":file.filename,"status":"error","message":str(e)})
    return {"results": results}

# ---------------------------
# checkin endpoint (multi-frame) using process_frames_consensus (which uses raw anti-spoof)
# ---------------------------
@app.post("/api/checkin")
async def checkin(files: List[UploadFile] = File(None)):
    if not files or len(files)==0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    frames=[]
    for f in files:
        try:
            b = await f.read()
            frames.append(b)
        except Exception as e:
            print(f"[ERROR] reading file {getattr(f,'filename',None)}: {e}")
    if len(frames) < 3:
        return JSONResponse({"success": False, "message":"Need at least 3 frames for verification."})
    res = process_frames_consensus(frames, min_frames_required=3, distance_threshold=0.5)
    if res.get("status") != "success":
        return JSONResponse({"success":False, "message": res.get("message","Verification failed")})
    student_id = res["student_id"]
    confidence = res.get("confidence",0.0)
    # mark present
    mark_student_present(student_id, date.today())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, "checkin", "Camera checkin - ensemble anti-spoof verified"))
    cur.execute("UPDATE trust_scores SET score = MIN(score + 3,100), streak = streak + 1 WHERE student_id = ?", (student_id,))
    conn.commit(); conn.close()
    # broadcast
    try:
        avatar_url = get_avatar_url_for_student(student_id)
        await manager.broadcast(json.dumps({"type":"presence","payload":{"student_id":student_id,"status":"Present","timestamp":datetime.now().isoformat(),"avatarUrl":avatar_url}}))
    except Exception as e:
        print("[WARN] broadcast failed:", e)
    return JSONResponse({"success":True,"student_id":student_id,"status":"present","confidence":confidence,"message":"Verification successful!"})

# ---------------------------
# WebSocket endpoint - apply ensemble anti-spoof per incoming frame (raw method)
# ---------------------------
@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({"type":"info","message":f"Recognition ready. Loaded faces: {len(known_names)}"}))
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
            except Exception as e:
                print("[WARN] ws receive error:", e)
                await asyncio.sleep(0.05)
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
                    b64 = b64.split(",",1)[1]
                try:
                    img_bytes = base64.b64decode(b64)
                except Exception:
                    await websocket.send_text(json.dumps({"type":"error","message":"invalid_base64"}))
                    continue
                if not known_encodings:
                    await websocket.send_text(json.dumps({"type":"error","message":"No registered faces loaded"}))
                    continue
                # convert to BGR
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    await websocket.send_text(json.dumps({"type":"error","message":"invalid_image"}))
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # locate faces
                try:
                    face_locs = face_recognition.face_locations(img_rgb)
                except Exception as e:
                    print("[WARN] face_locations failed:", e)
                    await websocket.send_text(json.dumps({"type":"error","message":"face detection failed"}))
                    continue
                if not face_locs:
                    await websocket.send_text(json.dumps({"type":"error","message":"No face detected"}))
                    continue
                # handle first face
                top,right,bottom,left = face_locs[0]
                bbox = (left, top, right, bottom)
                # anti-spoof ensemble check (raw)
                try:
                    if is_frame_suspicious_by_ensemble(img_bgr, bbox):
                        await websocket.send_text(json.dumps({"type":"suspicious","message":"Photo detected. Please show live face."}))
                        continue
                except Exception as e:
                    print("[WARN] ensemble check failed on websocket:", e)
                    await websocket.send_text(json.dumps({"type":"error","message":"anti-spoof failed"}))
                    continue
                # recognition
                enc = improved_encoding_from_bytes(img_bytes)
                if enc is None:
                    await websocket.send_text(json.dumps({"type":"error","message":"No face encoding"}))
                    continue
                try:
                    distances = face_recognition.face_distance(known_encodings, enc)
                except Exception as e:
                    print("[ERROR] face_distance failed:", e)
                    await websocket.send_text(json.dumps({"type":"error","message":"face matching failed"}))
                    continue
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx]) if len(distances)>0 else math.inf
                confidence = max(0.0, 1.0 - best_dist)
                WEBSOCKET_THRESHOLD = 0.55
                if best_dist <= WEBSOCKET_THRESHOLD:
                    student_id = known_names[best_idx]
                    mark_student_present(student_id, date.today())
                    conn = get_conn(); cur = conn.cursor()
                    cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, "checkin", "WebSocket checkin"))
                    cur.execute("UPDATE trust_scores SET score = MIN(score + 2,100), streak = streak+1 WHERE student_id = ?", (student_id,))
                    conn.commit(); conn.close()
                    await websocket.send_text(json.dumps({"type":"recognized","student_id":student_id,"confidence":confidence,"distance":best_dist,"message":f"Welcome {student_id}!"}))
                    # broadcast
                    try:
                        avatar_url = get_avatar_url_for_student(student_id)
                        await manager.broadcast(json.dumps({"type":"presence","payload":{"student_id":student_id,"status":"Present","timestamp":datetime.now().isoformat(),"avatarUrl":avatar_url}}))
                    except Exception as e:
                        print("[WARN] broadcast failed:", e)
                else:
                    debug_info = [f"{n}:{d:.3f}" for n,d in zip(known_names, distances)]
                    await websocket.send_text(json.dumps({"type":"unknown","message":f"Not recognized (best {best_dist:.3f})","debug":debug_info,"threshold":WEBSOCKET_THRESHOLD}))
            await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print("[ERROR] websocket main loop:", e)
        manager.disconnect(websocket)

# ---------------------------
# Additional endpoints: insights & attendance stats (kept simple)
# ---------------------------
@app.get("/api/insights")
async def api_insights():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT kind, text, impact, created_at FROM insights ORDER BY created_at DESC LIMIT 10")
    rows = cur.fetchall(); conn.close()
    return [dict(r) for r in rows]

@app.get("/api/attendance/stats")
async def attendance_stats():
    try:
        conn = get_conn(); cur = conn.cursor()
        today = date.today()
        cur.execute("""
            SELECT 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")t,
                SUM(CASE WHEN da.status = 'absent' THEN 1 ELSE 0 END) as absent_count
            FROM students s
            LEFT JOIN daily_attendance da ON s.id = da.student_id AND da.attendance_date = ?
        """, (today,))
        stats = dict(cur.fetchone()); conn.close()
        if stats["total_students"]>0:
            stats["present_percentage"] = round((stats["present_count"]/stats["total_students"])*100,1)
            stats["absent_percentage"] = round((stats["absent_count"]/stats["total_students"])*100,1)
        else:
            stats["present_percentage"]=stats["absent_percentage"]=0
        return stats
    except Exception as e:
        print("[ERROR] attendance_stats:", e)
        return {"total_students":0,"present_count":0,"absent_count":0,"present_percentage":0,"absent_percentage":0}

# ---------------------------
# Startup event - init DB and load faces
# ---------------------------
@app.on_event("startup")
async def startup_event():
    print("[INFO] Starting PresenceAI backend...")
    init_db()
    load_known_faces()
    print(f"[INFO] Loaded faces: {known_names}")
    if len(known_names)==0:
        print("[WARN] No faces loaded. Place images in known_faces directory named by student_id.jpg")

# ---------------------------
# RUN
# ---------------------------
if __name__ == "_main_":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")