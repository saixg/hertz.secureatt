from sqlmodel import create_engine, Session, select, SQLModel
from models import *
import json
from datetime import datetime, timedelta
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attendance.db")
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

def seed_data():
    """Seed the database with initial data for the 4 students"""
    with Session(engine) as session:
        # Check if data already exists
        existing_students = session.exec(select(Student)).all()
        if existing_students:
            print("Data already exists, skipping seed...")
            return

        # Create students
        students_data = [
            {"name": "Hasini", "avatar_url": "/avatars/hasini.jpg", "seat_row": 1, "seat_col": 1},
            {"name": "Anji", "avatar_url": "/avatars/anji.jpg", "seat_row": 1, "seat_col": 2},
            {"name": "Venkat", "avatar_url": "/avatars/venkat.jpg", "seat_row": 2, "seat_col": 1},
            {"name": "Sai Reddy", "avatar_url": "/avatars/sai_reddy.jpg", "seat_row": 2, "seat_col": 2}
        ]

        students = []
        for student_data in students_data:
            student = Student(**student_data)
            session.add(student)
            students.append(student)
        
        session.commit()
        session.refresh(students[0])  # Refresh to get IDs

        # Create trust scores for all students
        trust_scores_data = [
            {"student_id": 1, "score": 95.0, "punctuality": 98.0, "consistency": 92.0, "streak": 15},
            {"student_id": 2, "score": 87.0, "punctuality": 85.0, "consistency": 89.0, "streak": 8},
            {"student_id": 3, "score": 92.0, "punctuality": 90.0, "consistency": 94.0, "streak": 12},
            {"student_id": 4, "score": 78.0, "punctuality": 75.0, "consistency": 81.0, "streak": 3}
        ]

        for trust_data in trust_scores_data:
            trust_score = TrustScore(**trust_data)
            session.add(trust_score)

        # Create sample attendance events for the past week
        base_time = datetime.now() - timedelta(days=7)
        
        for day in range(7):
            day_time = base_time + timedelta(days=day)
            # Morning check-ins
            for student_id in range(1, 5):
                # Vary the timing to create realistic data
                if student_id == 1:  # Hasini - always on time
                    checkin_time = day_time.replace(hour=9, minute=0)
                    status = AttendanceStatus.PRESENT
                elif student_id == 2:  # Anji - sometimes late
                    checkin_time = day_time.replace(hour=9, minute=5 if day % 3 == 0 else 0)
                    status = AttendanceStatus.LATE if day % 3 == 0 else AttendanceStatus.PRESENT
                elif student_id == 3:  # Venkat - consistent
                    checkin_time = day_time.replace(hour=9, minute=2)
                    status = AttendanceStatus.PRESENT
                else:  # Sai Reddy - irregular
                    if day % 4 == 0:  # Absent sometimes
                        continue
                    checkin_time = day_time.replace(hour=9, minute=15 if day % 2 == 0 else 1)
                    status = AttendanceStatus.LATE if day % 2 == 0 else AttendanceStatus.PRESENT

                event = AttendanceEvent(
                    student_id=student_id,
                    timestamp=checkin_time,
                    event_type=EventType.CHECKIN,
                    status=status,
                    meta=json.dumps({"subject": "Computer Science", "room": "CS-101"})
                )
                session.add(event)

        # Create some insights
        insights_data = [
            {
                "kind": "trend",
                "text": "📈 Overall attendance improved by 12% this week compared to last week",
                "impact": "high"
            },
            {
                "kind": "anomaly", 
                "text": "🚨 3 suspicious face detections detected today - possible proxy attempts",
                "impact": "high"
            },
            {
                "kind": "highlight",
                "text": "🌟 Hasini maintains perfect punctuality record for 15 consecutive days",
                "impact": "med"
            },
            {
                "kind": "prediction",
                "text": "📊 Based on current trends, Monday morning attendance will likely drop by 8%",
                "impact": "med"
            },
            {
                "kind": "trend",
                "text": "⏰ Average late arrival time decreased from 12 minutes to 8 minutes this week",
                "impact": "low"
            }
        ]

        for insight_data in insights_data:
            insight = Insight(**insight_data)
            session.add(insight)

        session.commit()
        print("Database seeded successfully!")

def calculate_attendance_percentage(student_id: int) -> float:
    """Calculate attendance percentage for a student"""
    with Session(engine) as session:
        # Get total events for the student
        total_events = session.exec(
            select(AttendanceEvent).where(AttendanceEvent.student_id == student_id)
        ).all()
        
        if not total_events:
            return 0.0
            
        present_events = [e for e in total_events if e.status in [AttendanceStatus.PRESENT, AttendanceStatus.LATE]]
        return (len(present_events) / len(total_events)) * 100

def update_trust_score(student_id: int):
    """Update trust score based on recent attendance"""
    with Session(engine) as session:
        attendance_pct = calculate_attendance_percentage(student_id)
        
        trust_score = session.exec(
            select(TrustScore).where(TrustScore.student_id == student_id)
        ).first()
        
        if trust_score:
            trust_score.score = min(100.0, attendance_pct * 0.7 + trust_score.punctuality * 0.3)
            trust_score.updated_at = datetime.utcnow()
            session.add(trust_score)
            session.commit()

def get_current_status(student_id: int) -> AttendanceStatus:
    """Get current status for a student based on today's events"""
    with Session(engine) as session:
        today = datetime.now().date()
        today_events = session.exec(
            select(AttendanceEvent)
            .where(AttendanceEvent.student_id == student_id)
            .where(AttendanceEvent.timestamp >= datetime.combine(today, datetime.min.time()))
        ).all()
        
        if not today_events:
            return AttendanceStatus.ABSENT
            
        latest_event = max(today_events, key=lambda x: x.timestamp)
        return latest_event.status
