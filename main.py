# main.py
"""
PresenceAI - Attendance Backend (Corrected + attendance endpoint)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import sqlite3
import os
import json
import uuid
import secrets
import hashlib
import binascii
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import asyncio
import pathlib
import base64
import math
from datetime import date as _date

# Face libs (may require these installed; if not using face features you can stub)
try:
    import face_recognition
    import numpy as np
    import cv2
except Exception:
    face_recognition = None
    np = None
    cv2 = None

# Resolve base dir
BASE_DIR = pathlib.Path(__file__).resolve().parent

# Data dirs
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
AVATARS_DIR = BASE_DIR / "avatars"
TEMP_DIR = BASE_DIR / "temp"
for d in (KNOWN_FACES_DIR, AVATARS_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

# App init
app = FastAPI(title="PresenceAI Attendance Backend", version="1.0.0")

# Mount avatars folder at /avatars so frontend can access images directly
try:
    app.mount("/avatars", StaticFiles(directory=str(AVATARS_DIR)), name="avatars")
except Exception as e:
    print("[WARN] Could not mount /avatars:", e)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB
DB_PATH = str(BASE_DIR / "attendance.db")

# In-memory token store (demo)
TOKENS: Dict[str, str] = {}

# Known faces cache (if face_recognition loaded)
known_encodings: List = []
known_names: List[str] = []

# -------------------
# Password helpers
# -------------------
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

# -------------------
# DB helpers
# -------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # students table uses column name `class` to match existing schema
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

    # seed demo data if empty
    cur.execute("SELECT COUNT(*) FROM students")
    if cur.fetchone()[0] == 0:
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

    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        salt, hsh = hash_password("hodpass")
        cur.execute("INSERT INTO users (username, password_hash, salt, role, display_name) VALUES (?, ?, ?, ?, ?)",
                    ("hod", hsh, salt, "hod", "Head of Department"))
        salt, hsh = hash_password("teacher1pass")
        cur.execute("INSERT INTO users (username, password_hash, salt, role, display_name, assigned_classes) VALUES (?, ?, ?, ?, ?, ?)",
                    ("teacher1", hsh, salt, "teacher", "Mrs. Teacher", "A,B"))
        salt, hsh = hash_password("92460118732")
        cur.execute("INSERT INTO users (username, password_hash, salt, role, display_name, student_id) VALUES (?, ?, ?, ?, ?, ?)",
                    ("sai", hsh, salt, "parent", "Sai's Parent", "sai"))

    conn.commit()
    conn.close()
    print("[INFO] DB initialized / seeded")

# -------------------
# Token helpers
# -------------------
def create_token(username: str) -> str:
    token = uuid.uuid4().hex
    TOKENS[token] = username
    return token

def get_username_for_token(token: str) -> Optional[str]:
    return TOKENS.get(token)

def remove_token(token: str):
    TOKENS.pop(token, None)

# -------------------
# Auth dependency
# -------------------
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

# -------------------
# Face helpers (if available)
# -------------------
def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    if face_recognition is None:
        print("[INFO] face_recognition not available; skipping known faces load")
        return
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
    if face_recognition is None:
        return None
    try:
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            return None
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(img_rgb)
        if encs:
            return encs[0]
    except Exception as e:
        print("[ERROR] encoding_from_bytes", e)
    return None

# simple liveness heuristic
def basic_liveness_frame(data: bytes):
    if cv2 is None:
        return False
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return True
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        std = float(np.std(gray))
        if variance < 50 or std < 10:
            return True
        return False
    except Exception as e:
        print("[WARN] liveness check error", e)
        return False

# consensus multi-frame matching (safe)
def process_frames_consensus(frames_bytes: List[bytes], min_frames_required=2, distance_threshold=0.5):
    if face_recognition is None or not frames_bytes:
        return {"status": "error", "message": "Face recognition not available or no frames"}
    match_counts = defaultdict(int)
    confidences = defaultdict(list)
    suspicious_flag = False
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
        if best_dist <= distance_threshold:
            name = known_names[best_idx]
            match_counts[name] += 1
            confidences[name].append(confidence)
        if basic_liveness_frame(b):
            suspicious_flag = True
    if total_processed == 0:
        return {"status": "error", "message": "No faces detected in frames"}
    if not match_counts:
        return {"status": "error", "message": "Unknown person detected. Please register first."}
    best_name = max(match_counts.items(), key=lambda x: x[1])[0]
    count = match_counts[best_name]
    avg_conf = sum(confidences[best_name]) / (len(confidences[best_name]) or 1)
    if count >= min_frames_required or count >= (total_processed // 2 + 1):
        return {"status": "success", "student_id": best_name, "confidence": float(avg_conf), "is_suspicious": suspicious_flag}
    else:
        return {"status": "error", "message": "Could not confidently match the face. Try again."}

# -------------------
# WebSocket manager
# -------------------
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

# -------------------
# Avatar helper
# -------------------
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
        print("[WARN] get_avatar_url_for_student error", e)
    return "/avatars/default.jpg"

# -------------------
# API: health/login/me
# -------------------
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

# -------------------
# API: students list & detail
# -------------------
@app.get("/api/students")
async def api_students():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT s.id, s.name, s.avatar_url, s.seat_row, s.seat_col,
                   ts.score as trust_score,
                   CASE 
                       WHEN ae.ts IS NOT NULL AND ae.ts > datetime('now', '-5 minutes') THEN 
                           CASE WHEN ae.type = 'suspicious' THEN 'suspicious' ELSE 'present' END
                       WHEN ae.ts IS NOT NULL AND ae.ts > datetime('now', '-30 minutes') THEN 'late'
                       ELSE 'absent'
                   END as status,
                   ae.ts as last_checkin,
                   s.mobile, s.class
            FROM students s
            LEFT JOIN trust_scores ts ON s.id = ts.student_id
            LEFT JOIN (
                SELECT student_id, MAX(ts) as ts, type
                FROM attendance_events
                WHERE date(ts) = date('now')
                GROUP BY student_id
            ) ae ON s.id = ae.student_id
        """)
        rows = cur.fetchall()
        students = []
        for r in rows:
            students.append({
                "id": r["id"],
                "name": r["name"],
                "avatarUrl": r["avatar_url"] or "/avatars/default.jpg",
                "seat": {"row": r["seat_row"], "col": r["seat_col"]} if r["seat_row"] and r["seat_col"] else None,
                "trustScore": r["trust_score"] or 100,
                "status": (r["status"] or "absent").lower(),
                "smartTag": r["status"] or "Absent",
                "attendancePct": calculate_attendance_percentage(r["id"]),
                "liveSeenAt": r["last_checkin"],
                "mobile": r["mobile"],
                "class": r["class"]
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

# -------------------
# Internal register handler (used by endpoints)
# -------------------
async def _register_student_internal(student_id: str, name: str, seat_row: Optional[int], seat_col: Optional[int], mobile: Optional[str], class_name: Optional[str], faceImage: Optional[UploadFile], avatar: Optional[UploadFile]):
    conn = get_conn()
    cur = conn.cursor()

    avatar_url = None
    if avatar:
        avatar_path = AVATARS_DIR / f"{student_id}.jpg"
        with open(avatar_path, "wb") as f:
            f.write(await avatar.read())
        avatar_url = f"/avatars/{student_id}.jpg"
    else:
        existing_path = AVATARS_DIR / f"{student_id}.jpg"
        if existing_path.exists():
            avatar_url = f"/avatars/{student_id}.jpg"

    # Use DB column 'class'
    cur.execute("""
        INSERT OR REPLACE INTO students (id, name, avatar_url, seat_row, seat_col, mobile, class)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (student_id, name, avatar_url, seat_row, seat_col, mobile, class_name))

    cur.execute("""
        INSERT OR IGNORE INTO trust_scores (student_id, score, punctuality, consistency, streak)
        VALUES (?, 100, 100, 100, 0)
    """, (student_id,))

    if faceImage:
        face_path = KNOWN_FACES_DIR / f"{student_id}.jpg"
        with open(face_path, "wb") as f:
            f.write(await faceImage.read())

    conn.commit()
    conn.close()

    if faceImage:
        load_known_faces()

    return {"success": True, "student_id": student_id, "avatar_url": avatar_url}

# -------------------
# POST /api/students (alias for register)
# -------------------
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
    # HOD-only by design
    if user["role"] != "hod":
        raise HTTPException(status_code=403, detail="HOD required")
    return await _register_student_internal(student_id, name, seat_row, seat_col, mobile, class_name, face, avatar)

# -------------------
# DELETE student
# -------------------
@app.delete("/api/students/{student_id}")
async def delete_student(student_id: str, user = Depends(require_token)):
    if user["role"] != "hod":
        raise HTTPException(status_code=403, detail="HOD required")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM students WHERE id = ?", (student_id,))
    cur.execute("DELETE FROM trust_scores WHERE student_id = ?", (student_id,))
    cur.execute("DELETE FROM attendance_events WHERE student_id = ?", (student_id,))
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

# -------------------
# Mark attendance
# -------------------
@app.post("/api/attendance/mark")
async def mark_attendance(payload: Dict[str, Any], user = Depends(require_token)):
    student_id = payload.get("student_id")
    status = payload.get("status", "Present")
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id required")
    if user["role"] not in ("hod", "teacher"):
        raise HTTPException(status_code=403, detail="Teacher or HOD required")
    conn = get_conn()
    cur = conn.cursor()
    event_type = "suspicious" if status.lower() == "suspicious" else "checkin"
    label = f"Marked: {status}"
    cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, event_type, label))
    if status.lower() == "suspicious":
        cur.execute("UPDATE trust_scores SET score = MAX(score - 5, 0), updated_at = CURRENT_TIMESTAMP WHERE student_id = ?", (student_id,))
    else:
        cur.execute("UPDATE trust_scores SET score = MIN(score + 1, 100), streak = streak + 1, punctuality = MIN(punctuality + 1, 100), updated_at = CURRENT_TIMESTAMP WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()

    # broadcast presence
    try:
        avatar_url = get_avatar_url_for_student(student_id)
    except Exception:
        avatar_url = "/avatars/default.jpg"
    asyncio.create_task(manager.broadcast(json.dumps({
        "type": "presence",
        "payload": {"student_id": student_id, "status": status, "timestamp": datetime.now().isoformat(), "avatarUrl": avatar_url}
    })))
    return {"success": True}

# -------------------
# Checkin (multi-frame)
# -------------------
@app.post("/api/checkin")
async def checkin(files: List[UploadFile] = File(None)):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    frames = []
    for f in files:
        try:
            frames.append(await f.read())
        except:
            pass
    res = process_frames_consensus(frames, min_frames_required=2, distance_threshold=0.48)
    if res.get("status") != "success":
        return JSONResponse({"success": False, "message": res.get("message", "No match")})
    student_id = res["student_id"]
    is_suspicious = res["is_suspicious"]
    confidence = res.get("confidence", 0.0)
    conn = get_conn()
    cur = conn.cursor()
    try:
        if is_suspicious:
            cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, "suspicious", "Suspicious checkin"))
            cur.execute("UPDATE trust_scores SET score = MAX(score - 8, 0), updated_at = CURRENT_TIMESTAMP WHERE student_id = ?", (student_id,))
            status_label = "Suspicious"
        else:
            cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, "checkin", "Camera checkin"))
            cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, "present", "Auto-marked present via face scan"))
            cur.execute("UPDATE trust_scores SET score = MIN(score + 1, 100), streak = streak + 1, punctuality = MIN(punctuality + 1,100), updated_at = CURRENT_TIMESTAMP WHERE student_id = ?", (student_id,))
            status_label = "Present"
        conn.commit()
    finally:
        conn.close()

    try:
        avatar_url = get_avatar_url_for_student(student_id)
    except Exception:
        avatar_url = "/avatars/default.jpg"
    asyncio.create_task(manager.broadcast(json.dumps({
        "type": "presence",
        "payload": {"student_id": student_id, "status": status_label, "timestamp": datetime.now().isoformat(), "confidence": confidence, "avatarUrl": avatar_url}
    })))

    return {"success": True, "student_id": student_id, "status": status_label, "confidence": confidence}

# -------------------
# Timeline / trust / leaderboard / insights / seats
# -------------------
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

# -------------------
# New: GET /api/attendance
# -------------------
@app.get("/api/attendance")
async def api_attendance(date: Optional[str] = None, class_name: Optional[str] = None):
    """
    Returns attendance records for a specific date (YYYY-MM-DD).
    Optional query param: class_name (A/B/C or 'all')
    Response: [
      {
        "student_id": "sai",
        "name": "Sai",
        "class": "A",
        "status": "present"|"late"|"absent"|"suspicious",
        "timestamp": "2025-09-14T07:12:00",
        "avatarUrl": "/avatars/sai.jpg",
        "confidence": 0.92  # optional if recorded
      }, ...
    ]
    """
    try:
        # normalize date param (default to today)
        if not date:
            qdate = _date.today().isoformat()
        else:
            # allow both YYYY-MM-DD and other parseable strings
            try:
                # quick validation: accept YYYY-MM-DD
                _ = _date.fromisoformat(date)
                qdate = date
            except Exception:
                # fallback to today if parse fails
                qdate = _date.today().isoformat()

        conn = get_conn()
        cur = conn.cursor()

        # choose students filtered by class if requested
        if class_name and class_name.lower() != "all":
            cur.execute("SELECT id, name, avatar_url, class FROM students WHERE class = ?", (class_name,))
        else:
            cur.execute("SELECT id, name, avatar_url, class FROM students")
        studs = cur.fetchall()
        student_map = {r["id"]: {"id": r["id"], "name": r["name"], "avatarUrl": r["avatar_url"] or "/avatars/default.jpg", "class": r["class"]} for r in studs}

        # Load latest attendance_events for those students on the requested date
        # We'll select the most recent event per student on that date
        ids = tuple(student_map.keys())
        records = []
        if ids:
            # Compose query to get the latest event per student for that date
            cur.execute(f"""
                SELECT ae.student_id, ae.type, ae.ts, ae.label
                FROM attendance_events ae
                JOIN (
                  SELECT student_id, MAX(ts) as maxts FROM attendance_events
                  WHERE date(ts) = ?
                  GROUP BY student_id
                ) s2 ON ae.student_id = s2.student_id AND ae.ts = s2.maxts
                WHERE ae.student_id IN ({','.join('?' for _ in ids)})
            """, (qdate, *ids))
            rows = cur.fetchall()
            # Map by student_id
            att_by_student = {r["student_id"]: {"type": r["type"], "ts": r["ts"], "label": r["label"]} for r in rows}
        else:
            att_by_student = {}

        # Build response list using students list and any attendance found
        for sid, meta in student_map.items():
            att = att_by_student.get(sid)
            if att:
                typ = att.get("type") or ""
                # map event types -> status expected by frontend
                if typ.lower() in ("suspicious",):
                    status = "suspicious"
                elif typ.lower() in ("present","checkin"):
                    # if checkin timestamp older than threshold could be 'late', but frontend can treat late logic
                    status = "present"
                else:
                    status = att.get("type", "present").lower()
                rec = {
                    "student_id": sid,
                    "name": meta["name"],
                    "class": meta["class"],
                    "status": status,
                    "timestamp": att.get("ts"),
                    "avatarUrl": meta["avatarUrl"]
                }
            else:
                # no event for the date -> absent
                rec = {
                    "student_id": sid,
                    "name": meta["name"],
                    "class": meta["class"],
                    "status": "absent",
                    "timestamp": None,
                    "avatarUrl": meta["avatarUrl"]
                }
            records.append(rec)

        conn.close()
        return records
    except Exception as e:
        print("[ERROR] api_attendance", e)
        raise HTTPException(status_code=500, detail="attendance query failed")

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

# -------------------
# Simulate checkin (demo)
# -------------------
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
    event_type = "suspicious" if status.lower() == "suspicious" else "checkin"
    cur.execute("INSERT INTO attendance_events (student_id, type, label) VALUES (?, ?, ?)", (student_id, event_type, f"Simulated {status}"))
    if status.lower() == "suspicious":
        cur.execute("UPDATE trust_scores SET score = MAX(score - 5, 0) WHERE student_id = ?", (student_id,))
    else:
        cur.execute("UPDATE trust_scores SET score = MIN(score + 2, 100), streak = streak + 1 WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()
    try:
        avatar_url = get_avatar_url_for_student(student_id)
    except Exception:
        avatar_url = "/avatars/default.jpg"
    asyncio.create_task(manager.broadcast(json.dumps({"type":"presence","payload":{"student_id": student_id, "status": status, "timestamp": datetime.now().isoformat(), "avatarUrl": avatar_url}})))
    return {"success": True, "student_id": student_id, "student_name": name, "status": status}

# -------------------
# WebSocket endpoint
# -------------------
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

            # parse JSON if sent
            try:
                msg = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"type":"ack","message":"ok"}))
                continue

            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type":"ack","message":"ok"}))
                continue

            # handle base64 image payload
            if "img" in msg:
                b64 = msg.get("img") or ""
                if b64.startswith("data:") and "," in b64:
                    b64 = b64.split(",", 1)[1]
                try:
                    img_bytes = base64.b64decode(b64)
                except Exception:
                    await websocket.send_text(json.dumps({"type":"error", "message": "invalid_base64"}))
                    continue
                if face_recognition is None:
                    await websocket.send_text(json.dumps({"type":"error","message":"face_recognition_not_available"}))
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
                    resp = {"type":"recognized", "student_id": student_id, "confidence": confidence}
                    try:
                        avatar_url = get_avatar_url_for_student(student_id)
                    except Exception:
                        avatar_url = "/avatars/default.jpg"
                    asyncio.create_task(manager.broadcast(json.dumps({"type":"presence","payload":{"student_id": student_id, "status":"Present","timestamp": datetime.now().isoformat(), "confidence": confidence, "avatarUrl": avatar_url}})))
                else:
                    resp = {"type":"unknown", "best_distance": best_dist, "confidence": confidence}
                await websocket.send_text(json.dumps(resp))
                continue

            await websocket.send_text(json.dumps({"type":"ack","message":"ok"}))

    except Exception as e:
        print("[ERROR] ws_events outer", e)
        manager.disconnect(websocket)

# -------------------
# attendance % helper
# -------------------
def calculate_attendance_percentage(student_id: str) -> int:
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
          SELECT COUNT(DISTINCT DATE(ts)) as present_days
          FROM attendance_events
          WHERE student_id = ?
          AND type = 'checkin'
          AND ts >= datetime('now', '-30 days')
        """, (student_id,))
        pd = cur.fetchone()[0] or 0
        conn.close()
        return min(int((pd / 30) * 100), 100)
    except:
        return 85

# -------------------
# Startup
# -------------------
@app.on_event("startup")
async def startup():
    print("[INFO] Starting PresenceAI backend...")
    print("[INFO] BASE_DIR:", BASE_DIR)
    print("[INFO] avatars dir:", AVATARS_DIR.resolve())
    try:
        print("[INFO] avatars contents:", [p.name for p in AVATARS_DIR.iterdir()])
    except Exception as e:
        print("[WARN] avatars listing failed:", e)
    init_db()
    load_known_faces()
    print(f"[INFO] known faces loaded: {len(known_names)}")

# -------------------
# Serve avatar file fallback (optional)
# -------------------
@app.get("/avatars/{filename}")
async def avatar_file(filename: str):
    filename = os.path.basename(filename)
    path = AVATARS_DIR / filename
    if path.exists() and path.is_file():
        return FileResponse(str(path))
    default = AVATARS_DIR / "default.jpg"
    if default.exists() and default.is_file():
        return FileResponse(str(default))
    raise HTTPException(status_code=404, detail="Avatar not found")

# -------------------
# Run
# -------------------
if __name__ == "__main__":
    init_db()
    load_known_faces()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
