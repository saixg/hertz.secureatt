from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional, List
from datetime import datetime
from enum import Enum
import json  # ✅ This is fine to keep, even if it's not used in this file

# Enums
class AttendanceStatus(str, Enum):
    PRESENT = "Present"
    ABSENT = "Absent"
    LATE = "Late"
    SUSPICIOUS = "Suspicious"

class EventType(str, Enum):
    CHECKIN = "checkin"
    CLASS = "class"
    ABSENT = "absent"
    CHECKOUT = "checkout"

# Database Models
class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    avatar_url: Optional[str] = None
    seat_row: Optional[int] = None
    seat_col: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FaceEncoding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id")
    encoding_vector: str  # JSON string of the encoding array
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AttendanceEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: EventType
    status: AttendanceStatus = AttendanceStatus.PRESENT
    meta: Optional[str] = None  # JSON string for additional data
    confidence: Optional[float] = None

class TrustScore(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id", unique=True)
    score: float = Field(default=85.0)
    punctuality: float = Field(default=85.0)
    consistency: float = Field(default=85.0)
    streak: int = Field(default=0)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class LeaderboardSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    metric: str  # "punctuality", "consistency", "overall"
    week_start: datetime
    rows_json: str  # JSON string of leaderboard data
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Insight(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str  # "trend", "anomaly", "highlight", "prediction"
    text: str
    impact: str = Field(default="low")  # "low", "med", "high"
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Response Models
class StudentResponse(SQLModel):
    id: int
    name: str
    avatar_url: Optional[str] = None
    seat: Optional[dict] = None
    status: AttendanceStatus = AttendanceStatus.ABSENT
    smart_tag: Optional[str] = None
    attendance_pct: Optional[float] = None
    trust_score: Optional[float] = None
    live_seen_at: Optional[datetime] = None

class AttendanceEventResponse(SQLModel):
    id: int
    student_id: int
    timestamp: datetime
    event_type: EventType
    status: AttendanceStatus
    label: str
    meta: Optional[dict] = None

class TrustScoreResponse(SQLModel):
    student_id: int
    score: float
    punctuality: float
    consistency: float
    streak: int

class LeaderboardRow(SQLModel):
    id: int
    name: str
    avatar_url: Optional[str] = None
    score: float
    trend: int  # Change from last week

class InsightResponse(SQLModel):
    id: int
    kind: str
    text: str
    created_at: datetime
    impact: str

class CheckinRequest(SQLModel):
    image_data: str  # base64 encoded image

class CheckinResponse(SQLModel):
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    status: AttendanceStatus
    confidence: float
    message: str  # ✅ This was previously incorrect (you had `message: str as`) — now fixed
 