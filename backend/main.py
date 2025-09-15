from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
import base64
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import csv
import face_recognition
import cv2
import numpy as np
from PIL import Image

app = FastAPI(title="Attendance System Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories
os.makedirs("known_faces", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Load known faces
known_encodings = []
known_names = []

def load_known_faces():
    """Load known faces from known_faces directory"""
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        return
    
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                path = os.path.join(known_faces_dir, filename)
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                
                if encodings:
                    encoding = encodings[0]
                    known_encodings.append(encoding)
                    # Use filename without extension as student ID
                    name = os.path.splitext(filename)[0]
                    known_names.append(name)
                    print(f"[INFO] Loaded face encoding for: {name}")
                else:
                    print(f"[WARNING] No face found in: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to load {filename}: {e}")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            pass

    async def broadcast(self, message: str):
        for connection in self.active_connections[:]:  # Create a copy of the list
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Database initialization
def init_db():
    """Initialize database with all tables"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            avatar_url TEXT,
            seat_row INTEGER,
            seat_col INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            type TEXT NOT NULL,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            label TEXT,
            subject TEXT,
            room TEXT,
            note TEXT,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    
    # Trust scores table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trust_scores (
            student_id TEXT PRIMARY KEY,
            score INTEGER DEFAULT 100,
            punctuality INTEGER DEFAULT 100,
            consistency INTEGER DEFAULT 100,
            streak INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    
    # Insights table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            impact TEXT DEFAULT 'low'
        )
    ''')
    
    # Leaderboard snapshots table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric TEXT NOT NULL,
            week_start DATE NOT NULL,
            rows_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    
    # Check if we need to seed data
    cursor.execute("SELECT COUNT(*) FROM students")
    student_count = cursor.fetchone()[0]
    
    if student_count == 0:
        seed_database(cursor, conn)
    
    conn.close()
    print("[INFO] Database initialized successfully")

def seed_database(cursor, conn):
    """Seed database with your actual students"""
    print("[INFO] Seeding database with students...")
    
    # Your actual students - will be updated based on images found
    students = [
        ("sai", "Sai", "/avatars/sai.jpg", 1, 1),
        ("image_person", "Image Person", "/avatars/image_person.jpg", 1, 2),
    ]
    
    for student_id, name, avatar, row, col in students:
        cursor.execute("""
            INSERT OR REPLACE INTO students (id, name, avatar_url, seat_row, seat_col)
            VALUES (?, ?, ?, ?, ?)
        """, (student_id, name, avatar, row, col))
        
        # Initialize trust scores
        cursor.execute("""
            INSERT OR REPLACE INTO trust_scores (student_id, score, punctuality, consistency, streak)
            VALUES (?, 100, 100, 100, 0)
        """, (student_id,))
    
    # Sample insights
    insights = [
        ("trend", "📈 Attendance system initialized and ready!", "high"),
        ("highlight", "🌟 Face recognition system active", "high"),
        ("prediction", "📊 Ready to track daily attendance patterns", "med"),
        ("anomaly", "🔍 Anti-spoofing detection enabled", "med")
    ]
    
    cursor.executemany("""
        INSERT INTO insights (kind, text, impact) VALUES (?, ?, ?)
    """, insights)
    
    conn.commit()
    print(f"[INFO] Added {len(students)} students and {len(insights)} insights")

def process_face_image(image_path):
    """Process uploaded image for face recognition"""
    try:
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            return {"status": "error", "message": "Could not load image"}
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_locations:
            return {"status": "error", "message": "No face detected in image"}
        
        if len(face_locations) > 1:
            return {"status": "error", "message": "Multiple faces detected. Please ensure only one person is in frame"}
        
        # Process the single face
        face_encoding = face_encodings[0]
        
        # Compare with known faces
        student_id = "Unknown"
        student_name = "Unknown"
        is_recognized = False
        is_suspicious = False
        
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if True in matches:
                best_match = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match]
                
                if confidence > 0.4:  # Lower threshold for better matching
                    student_id = known_names[best_match]
                    student_name = get_student_name_from_db(student_id)
                    is_recognized = True
                    
                    # Basic liveness check
                    is_suspicious = basic_liveness_check(frame, face_locations[0])
        
        if not is_recognized:
            return {
                "status": "error",
                "message": "Unknown person detected. Please register your face first."
            }
        
        # Return result
        message = f"Recognized: {student_name}"
        if is_suspicious:
            message += " (Suspicious - possible spoofing detected)"
        
        return {
            "status": "success",
            "student_id": student_id,
            "student_name": student_name,
            "is_suspicious": is_suspicious,
            "message": message
        }
        
    except Exception as e:
        print(f"[ERROR] Error processing image: {e}")
        return {"status": "error", "message": f"Processing error: {str(e)}"}

def basic_liveness_check(frame, face_location):
    """Basic liveness check without specialized models"""
    try:
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]
        
        if face_img.size == 0:
            return True  # Suspicious if no face image
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate image variance - printed photos tend to have lower variance
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check brightness patterns
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Simple heuristics
        if variance < 50 or brightness_std < 15:
            return True  # Suspicious
        
        return False
        
    except Exception as e:
        print(f"[WARNING] Liveness check failed: {e}")
        return False

def get_student_name_from_db(student_id):
    """Get student name from database"""
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else student_id.replace('_', ' ').title()
    except:
        return student_id.replace('_', ' ').title()

def calculate_attendance_percentage(student_id: str) -> int:
    """Calculate attendance percentage for last 30 days"""
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT DATE(ts)) as present_days
            FROM attendance_events 
            WHERE student_id = ? 
            AND type = 'checkin' 
            AND ts >= datetime('now', '-30 days')
        """, (student_id,))
        
        present_days = cursor.fetchone()[0] or 0
        conn.close()
        
        # Assume 30 possible days, calculate percentage
        return min(int((present_days / 30) * 100), 100)
    except:
        return 85  # Default value

def get_smart_tag(status: str, last_checkin: str) -> str:
    """Generate smart tag based on status"""
    if status == "Present":
        return "🟢 On Time"
    elif status == "Late":
        return "🟠 Late arrival"
    elif status == "Suspicious":
        return "🔴 Suspicious activity"
    else:
        return "⚪ Not present"

# Health check
@app.get("/api/healthz")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Students endpoints
@app.get("/api/students")
async def get_students():
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.name, s.avatar_url, s.seat_row, s.seat_col,
                   ts.score as trust_score,
                   CASE 
                       WHEN ae.ts IS NOT NULL AND ae.ts > datetime('now', '-5 minutes') THEN 
                           CASE WHEN ae.type = 'suspicious' THEN 'Suspicious' ELSE 'Present' END
                       WHEN ae.ts IS NOT NULL AND ae.ts > datetime('now', '-30 minutes') THEN 'Late'
                       ELSE 'Absent'
                   END as status,
                   ae.ts as last_checkin
            FROM students s
            LEFT JOIN trust_scores ts ON s.id = ts.student_id
            LEFT JOIN (
                SELECT student_id, MAX(ts) as ts, type
                FROM attendance_events 
                WHERE date(ts) = date('now')
                GROUP BY student_id
            ) ae ON s.id = ae.student_id
        """)
        
        students = []
        for row in cursor.fetchall():
            student = {
                "id": row[0],
                "name": row[1],
                "avatarUrl": row[2] or f"/avatars/default.jpg",
                "seat": {"row": row[3], "col": row[4]} if row[3] and row[4] else None,
                "trustScore": row[5] or 100,
                "status": row[6] or "Absent",
                "smartTag": get_smart_tag(row[6] or "Absent", row[7]),
                "attendancePct": calculate_attendance_percentage(row[0]),
                "liveSeenAt": row[7]
            }
            students.append(student)
        
        conn.close()
        return students
        
    except Exception as e:
        print(f"[ERROR] Error getting students: {e}")
        return []

# Check-in endpoint
@app.post("/api/checkin")
async def checkin(file: UploadFile = File(...)):
    try:
        # Read image
        image_data = await file.read()
        
        # Save temporary file for processing
        temp_path = f"temp/temp_{datetime.now().timestamp()}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)
        
        # Process with face recognition
        result = process_face_image(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if result["status"] == "success":
            # Record attendance
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            
            event_type = "suspicious" if result["is_suspicious"] else "checkin"
            label = "Suspicious check-in detected" if result["is_suspicious"] else "Check-in successful"
            
            cursor.execute("""
                INSERT INTO attendance_events (student_id, type, label)
                VALUES (?, ?, ?)
            """, (result["student_id"], event_type, label))
            
            # Update trust score
            if result["is_suspicious"]:
                cursor.execute("""
                    UPDATE trust_scores 
                    SET score = MAX(score - 10, 0),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE student_id = ?
                """, (result["student_id"],))
            else:
                cursor.execute("""
                    UPDATE trust_scores 
                    SET score = MIN(score + 1, 100),
                        streak = streak + 1,
                        punctuality = MIN(punctuality + 1, 100),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE student_id = ?
                """, (result["student_id"],))
            
            conn.commit()
            conn.close()
            
            # Broadcast update via WebSocket
            await manager.broadcast(json.dumps({
                "type": "presence",
                "payload": {
                    "student_id": result["student_id"],
                    "status": "Suspicious" if result["is_suspicious"] else "Present",
                    "timestamp": datetime.now().isoformat()
                }
            }))
            
            return {
                "success": True,
                "student_id": result["student_id"],
                "student_name": result["student_name"],
                "status": "Suspicious" if result["is_suspicious"] else "Present",
                "message": result["message"]
            }
        else:
            return {"success": False, "message": result["message"]}
            
    except Exception as e:
        print(f"[ERROR] Check-in error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Timeline endpoint
@app.get("/api/timeline/{student_id}")
async def get_timeline(student_id: str):
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, type, ts, label, subject, room, note
            FROM attendance_events
            WHERE student_id = ?
            ORDER BY ts DESC
            LIMIT 50
        """, (student_id,))
        
        events = []
        for row in cursor.fetchall():
            event = {
                "id": str(row[0]),
                "studentId": student_id,
                "type": row[1],
                "ts": row[2],
                "label": row[3],
                "meta": {
                    "subject": row[4],
                    "room": row[5],
                    "note": row[6]
                }
            }
            events.append(event)
        
        conn.close()
        return events
        
    except Exception as e:
        print(f"[ERROR] Timeline error: {e}")
        return []

# Trust score endpoint
@app.get("/api/trust/{student_id}")
async def get_trust_score(student_id: str):
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT score, punctuality, consistency, streak
            FROM trust_scores
            WHERE student_id = ?
        """, (student_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {"studentId": student_id, "score": 100, "punctuality": 100, "consistency": 100, "streak": 0}
        
        return {
            "studentId": student_id,
            "score": row[0],
            "punctuality": row[1],
            "consistency": row[2],
            "streak": row[3]
        }
        
    except Exception as e:
        print(f"[ERROR] Trust score error: {e}")
        return {"studentId": student_id, "score": 100, "punctuality": 100, "consistency": 100, "streak": 0}

# Leaderboard endpoint
@app.get("/api/leaderboard")
async def get_leaderboard(metric: str = "overall"):
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        if metric == "punctuality":
            order_col = "ts.punctuality"
        elif metric == "consistency":
            order_col = "ts.consistency"
        else:
            order_col = "ts.score"
        
        cursor.execute(f"""
            SELECT s.id, s.name, s.avatar_url, {order_col} as score, 
                   0 as trend
            FROM students s
            LEFT JOIN trust_scores ts ON s.id = ts.student_id
            ORDER BY score DESC
            LIMIT 10
        """)
        
        leaderboard = []
        for row in cursor.fetchall():
            entry = {
                "id": row[0],
                "name": row[1],
                "avatarUrl": row[2],
                "score": row[3] or 100,
                "trend": row[4]
            }
            leaderboard.append(entry)
        
        conn.close()
        return leaderboard
        
    except Exception as e:
        print(f"[ERROR] Leaderboard error: {e}")
        return []

# Insights endpoint
@app.get("/api/insights")
async def get_insights(role: str = "teacher"):
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, kind, text, created_at, impact
            FROM insights
            ORDER BY created_at DESC
            LIMIT 20
        """)
        
        insights = []
        for row in cursor.fetchall():
            insight = {
                "id": str(row[0]),
                "kind": row[1],
                "text": row[2],
                "createdAt": row[3],
                "impact": row[4]
            }
            insights.append(insight)
        
        conn.close()
        return insights
        
    except Exception as e:
        print(f"[ERROR] Insights error: {e}")
        return []

# Seats endpoint
@app.get("/api/seats")
async def get_seats():
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, seat_row, seat_col
            FROM students
            WHERE seat_row IS NOT NULL AND seat_col IS NOT NULL
        """)
        
        seats = {}
        for row in cursor.fetchall():
            key = f"{row[2]}-{row[3]}"
            seats[key] = {
                "studentId": row[0],
                "studentName": row[1],
                "row": row[2],
                "col": row[3]
            }
        
        conn.close()
        return seats
        
    except Exception as e:
        print(f"[ERROR] Seats error: {e}")
        return {}

# Simulate check-in endpoint (for demo/fallback)
@app.post("/api/simulate-checkin")
async def simulate_checkin(data: dict):
    try:
        student_id = data.get("student_id")
        status = data.get("status", "Present")
        
        if not student_id:
            raise HTTPException(status_code=400, detail="Student ID required")
        
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Get student name
        cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            raise HTTPException(status_code=404, detail="Student not found")
        
        student_name = result[0]
        
        # Record event
        event_type = "suspicious" if status == "Suspicious" else "checkin"
        label = f"Simulated {status.lower()}"
        
        cursor.execute("""
            INSERT INTO attendance_events (student_id, type, label)
            VALUES (?, ?, ?)
        """, (student_id, event_type, label))
        
        # Update trust score
        if status == "Suspicious":
            cursor.execute("""
                UPDATE trust_scores 
                SET score = MAX(score - 5, 0)
                WHERE student_id = ?
            """, (student_id,))
        else:
            cursor.execute("""
                UPDATE trust_scores 
                SET score = MIN(score + 2, 100),
                    streak = streak + 1
                WHERE student_id = ?
            """, (student_id,))
        
        conn.commit()
        conn.close()
        
        # Broadcast update
        await manager.broadcast(json.dumps({
            "type": "presence",
            "payload": {
                "student_id": student_id,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        }))
        
        return {
            "success": True,
            "student_id": student_id,
            "student_name": student_name,
            "status": status,
            "message": f"Simulated {status.lower()} for {student_name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Simulate check-in error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup"""
    print("[INFO] Starting Attendance System Backend...")
    print("[INFO] Initializing database...")
    init_db()
    print("[INFO] Loading known faces...")
    load_known_faces()
    print(f"[INFO] Loaded {len(known_names)} known faces")
    print("[INFO] Backend ready!")

if __name__ == "__main__":
    # Initialize everything
    init_db()
    load_known_faces()
    
    print("🚀 Starting Attendance System Backend...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔗 WebSocket: ws://localhost:8000/ws/events")
    print("👥 Students API: http://localhost:8000/api/students")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )