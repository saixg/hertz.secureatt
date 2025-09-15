#!/usr/bin/env python3
"""
Test Your Attendance System
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_server():
    """Test if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/healthz")
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to run: python main.py")
        return False

def test_students():
    """Test students endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/students")
        if response.status_code == 200:
            students = response.json()
            print(f"✅ Found {len(students)} students:")
            for student in students:
                print(f"   • {student['name']} ({student['id']}) - {student.get('status', 'Absent')}")
            return students
        else:
            print(f"❌ Students endpoint failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_simulate_checkins():
    """Test simulate check-in for your students"""
    your_students = [("797", "Sai"), ("798", "Hasini"), ("833", "Venkat")]

    print("\n🎭 Testing simulate check-ins:")
    for student_id, student_name in your_students:
        data = {"student_id": student_id, "status": "Present"}
        try:
            response = requests.post(f"{BASE_URL}/api/simulate-checkin", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {student_name}: {result['message']}")
            else:
                print(f"   ❌ {student_name}: Failed ({response.status_code})")
        except Exception as e:
            print(f"   ❌ {student_name}: Error - {e}")

def main():
    print("🧪 TESTING YOUR ATTENDANCE SYSTEM")
    print("=" * 50)

    if not test_server():
        return

    print("\n👥 Testing students...")
    test_students()

    print("\n🎭 Testing check-ins...")
    test_simulate_checkins()

    print("\n🌐 Next Steps:")
    print("1. Open Swagger UI: http://localhost:8000/docs")
    print("2. Try GET /api/students")
    print("3. Try POST /api/simulate-checkin")
    print("4. For real face recognition, use POST /api/checkin with image upload")

    print("\n✅ Your backend is ready for the frontend team!")

if __name__ == "__main__":
    main()
