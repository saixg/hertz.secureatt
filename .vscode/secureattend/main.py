from fastapi import FastAPI, Depends, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from typing import List, Optional
import json
from datetime import datetime, timedelta

# Import our modules (make sure these files are in the same directory)
from models import *
from database import get_session, create_db_and_tables, seed_data, calculate_attendance_percentage, get_current_status
from ml_service import process_checkin_frame
from websocket_manager import manager, websocket_endpoint

# Create FastAPI app
app = FastAPI(
    title="SecureAttend API",
    description="Real-time attendance system with face recognition and anti-spoofing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    seed_data()

# Health check endpoint
@app.get("/healthz")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow()}

# WebSocket endpoint
@app.websocket("/events")
async def websocket_events_endpoint(websocket: WebSocket):
    await websocket_endpoint(websocket)

# Student endpoints
@app.get("/api/students", response_model=List[StudentResponse])
def get_students(session: Session = Depends(get_session)):
    """Get all students with their current status and stats"""
    students = session.exec(select(Student)).all()
    
    result = []
    for student in students:
        # Get current status
        status = get_current_status(student.id)
        
        # Get attendance percentage
        attendance_pct = calculate_attendance_percentage(student.id)
        
        # Get trust score
        trust_score_record = session.exec(
            select(TrustScore).where(TrustScore.student_id == student.id)
        ).first()
        trust_score = trust_score_record.score if trust_score_record else 85.0
        
        # Generate smart tag based on status
        smart_tag = None
        if status == AttendanceStatus.PRESENT:
            smart_tag = "🟢 On Time"
        elif status == AttendanceStatus.LATE:
            smart_tag = "🟠 Late"  
        elif status == AttendanceStatus.SUSPICIOUS:
            smart_tag = "🔴 Suspicious Face"
        else:
            smart_tag = "⚪ Absent"
        
        # Create seat info
        seat = None
        if student.seat_row and student.seat_col:
            seat = {"row": student.seat_row, "col": student.seat_col}
        
        result.append(StudentResponse(
            id=student.id,
            name=student.name,
            avatar_url=student.avatar_url,
            seat=seat,
            status=status,
            smart_tag=smart_tag,
            attendance_pct=attendance_pct,
            trust_score=trust_score,
            live_seen_at=datetime.utcnow() if status == AttendanceStatus.PRESENT else None
        ))
    
    return result

@app.get("/students/{student_id}", response_model=StudentResponse)
def get_student(student_id: int, session: Session = Depends(get_session)):
    """Get a specific student"""
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Same logic as get_students but for single student
    status = get_current_status(student.id)
    attendance_pct = calculate_attendance_percentage(student.id)
    
    trust_score_record = session.exec(
        select(TrustScore).where(TrustScore.student_id == student.id)
    ).first()
    trust_score = trust_score_record.score if trust_score_record else 85.0
    
    smart_tag = None
    if status == AttendanceStatus.PRESENT:
        smart_tag = "🟢 On Time"
    elif status == AttendanceStatus.LATE:
        smart_tag = "🟠 Late"
    elif status == AttendanceStatus.SUSPICIOUS:
        smart_tag = "🔴 Suspicious Face"
    else:
        smart_tag = "⚪ Absent"
    
    seat = None
    if student.seat_row and student.seat_col:
        seat = {"row": student.seat_row, "col": student.seat_col}
    
    return StudentResponse(
        id=student.id,
        name=student.name,
        avatar_url=student.avatar_url,
        seat=seat,
        status=status,
        smart_tag=smart_tag,
        attendance_pct=attendance_pct,
        trust_score=trust_score,
        live_seen_at=datetime.utcnow() if status == AttendanceStatus.PRESENT else None
    )

# Timeline endpoints
@app.get("/timeline/{student_id}", response_model=List[AttendanceEventResponse])
def get_student_timeline(student_id: int, session: Session = Depends(get_session)):
    """Get timeline events for a specific student"""
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Get events for the student, ordered by timestamp desc
    events = session.exec(
        select(AttendanceEvent)
        .where(AttendanceEvent.student_id == student_id)
        .order_by(AttendanceEvent.timestamp.desc())
    ).all()
    
    result = []
    for event in events:
        # Generate label based on event type and status
        if event.event_type == EventType.CHECKIN:
            if event.status == AttendanceStatus.PRESENT:
                label = f"✅ Checked in on time"
            elif event.status == AttendanceStatus.LATE:
                label = f"🟠 Checked in late"
            elif event.status == AttendanceStatus.SUSPICIOUS:
                label = f"🚨 Suspicious check-in attempt"
            else:
                label = f"📝 Check-in recorded"
        elif event.event_type == EventType.CHECKOUT:
            label = f"🏃 Checked out"
        elif event.event_type == EventType.CLASS:
            label = f"📚 Class session"
        else:
            label = f"❌ Marked absent"
        
        # Parse meta data
        meta = None
        if event.meta:
            try:
                meta = json.loads(event.meta)
            except:
                meta = None
        
        result.append(AttendanceEventResponse(
            id=event.id,
            student_id=event.student_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            status=event.status,
            label=label,
            meta=meta
        ))
    
    return result

# Trust score endpoints
@app.get("/trust/{student_id}", response_model=TrustScoreResponse)
def get_trust_score(student_id: int, session: Session = Depends(get_session)):
    """Get trust score for a specific student"""
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    trust_score = session.exec(
        select(TrustScore).where(TrustScore.student_id == student_id)
    ).first()
    
    if not trust_score:
        # Create default trust score if not exists
        trust_score = TrustScore(
            student_id=student_id,
            score=85.0,
            punctuality=85.0,
            consistency=85.0,
            streak=0
        )
        session.add(trust_score)
        session.commit()
        session.refresh(trust_score)
    
    return TrustScoreResponse(
        student_id=trust_score.student_id,
        score=trust_score.score,
        punctuality=trust_score.punctuality,
        consistency=trust_score.consistency,
        streak=trust_score.streak
    )

# Leaderboard endpoints
@app.get("/leaderboard", response_model=List[LeaderboardRow])
def get_leaderboard(metric: str = "overall", session: Session = Depends(get_session)):
    """Get leaderboard for a specific metric"""
    # Get all students with their trust scores
    query = session.exec(
        select(Student, TrustScore)
        .join(TrustScore, Student.id == TrustScore.student_id)
    ).all()
    
    # Sort based on metric
    leaderboard_data = []
    for student, trust_score in query:
        if metric == "punctuality":
            score = trust_score.punctuality
        elif metric == "consistency":
            score = trust_score.consistency
        else:  # overall
            score = trust_score.score
        
        leaderboard_data.append({
            "student": student,
            "score": score
        })
    
    # Sort by score descending
    leaderboard_data.sort(key=lambda x: x["score"], reverse=True)
    
    # Convert to response format with mock trends
    result = []
    for i, data in enumerate(leaderboard_data):
        student = data["student"]
        score = data["score"]
        
        # Mock trend calculation (you can make this more sophisticated)
        trend = 2 if i < 2 else (-1 if i >= len(leaderboard_data) - 2 else 0)
        
        result.append(LeaderboardRow(
            id=student.id,
            name=student.name,
            avatar_url=student.avatar_url,
            score=score,
            trend=trend
        ))
    
    return result

# Insights endpoints
@app.get("/insights", response_model=List[InsightResponse])
def get_insights(role: str = "teacher", session: Session = Depends(get_session)):
    """Get insights based on role"""
    # Get all insights, you can filter by role if needed
    insights = session.exec(
        select(Insight).order_by(Insight.created_at.desc()).limit(10)
    ).all()
    
    result = []
    for insight in insights:
        result.append(InsightResponse(
            id=insight.id,
            kind=insight.kind,
            text=insight.text,
            created_at=insight.created_at,
            impact=insight.impact
        ))
    
    return result

# Seats endpoint
@app.get("/seats")
def get_seats(session: Session = Depends(get_session)):
    """Get seating arrangement"""
    students = session.exec(select(Student)).all()
    
    # Create a 2x2 grid for the 4 students
    seats = {}
    for student in students:
        if student.seat_row and student.seat_col:
            key = f"{student.seat_row}-{student.seat_col}"
            seats[key] = {
                "student_id": student.id,
                "student_name": student.name,
                "row": student.seat_row,
                "col": student.seat_col,
                "status": get_current_status(student.id).value
            }
    
    return {"seats": seats, "grid_size": {"rows": 2, "cols": 2}}

# Check-in endpoint (ML integration)
@app.post("/api/checkin", response_model=CheckinResponse)
async def checkin(request: CheckinRequest, session: Session = Depends(get_session)):
    """Process check-in with face recognition and anti-spoofing"""
    try:
        # Process the image using ML service
        result = process_checkin_frame(request.image_data)
        
        # Create attendance event if student is recognized
        if result["student_id"] and result["status"] in ["Present", "Late"]:
            # Check if already checked in today
            today = datetime.now().date()
            existing_event = session.exec(
                select(AttendanceEvent)
                .where(AttendanceEvent.student_id == result["student_id"])
                .where(AttendanceEvent.timestamp >= datetime.combine(today, datetime.min.time()))
                .where(AttendanceEvent.event_type == EventType.CHECKIN)
            ).first()
            
            if not existing_event:
                # Create new attendance event
                event = AttendanceEvent(
                    student_id=result["student_id"],
                    event_type=EventType.CHECKIN,
                    status=AttendanceStatus(result["status"]),
                    confidence=result["confidence"],
                    meta=json.dumps({"method": "face_recognition", "message": result["message"]})
                )
                session.add(event)
                session.commit()
                
                # Broadcast update via WebSocket
                await manager.broadcast_presence_update(
                    student_id=result["student_id"],
                    status=result["status"],
                    student_name=result["student_name"]
                )
                
                # Broadcast timeline update
                await manager.broadcast_timeline_update(
                    student_id=result["student_id"],
                    event_data={
                        "id": event.id,
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type.value,
                        "status": event.status.value,
                        "label": f"✅ Checked in - {result['message']}"
                    }
                )
        
        return CheckinResponse(
            student_id=result["student_id"],
            student_name=result["student_name"], 
            status=AttendanceStatus(result["status"]),
            confidence=result["confidence"],
            message=result["message"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Check-in processing failed: {str(e)}")

# Simulate endpoints for testing without camera
@app.post("/simulate/checkin/{student_id}")
async def simulate_checkin(student_id: int, status: str, session: Session = Depends(get_session)):
    """Simulate a check-in for testing purposes"""
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    if status not in ["Present", "Late", "Suspicious"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    # Create attendance event
    event = AttendanceEvent(
        student_id=student_id,
        event_type=EventType.CHECKIN,
        status=AttendanceStatus(status),
        confidence=1.0,
        meta=json.dumps({"method": "simulation"})
    )
    session.add(event)
    session.commit()
    
    # Broadcast updates
    await manager.broadcast_presence_update(
        student_id=student_id,
        status=status,
        student_name=student.name
    )
    
    return {"message": f"Simulated {status} check-in for {student.name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)