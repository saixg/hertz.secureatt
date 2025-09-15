#!/usr/bin/env python3
"""
Setup Your Images for Attendance System
This copies your sai.jpg and image.png to the right place
"""

import os
import shutil
import sqlite3
from PIL import Image


def setup_your_images():
    """Setup your specific images"""
    print("📸 Setting up your images...")

    # Ensure known_faces directory exists
    os.makedirs("known_faces", exist_ok=True)

    # Your current images and where they should go
    image_mappings = [
    ("sai.jpg", "sai.jpg", "797", "Sai"),
    ("hasini.jpg", "hasini.jpg", "798", "Hasini"),
    ("me.jpg", "me.jpg", "833", "Venkat")
]



    setup_students = []

    for source_file, target_file, student_id, student_name in image_mappings:
        if os.path.exists(source_file):
            print(f"   ✅ Found {source_file}")

            target_path = os.path.join("known_faces", target_file)

            # Convert PNG to JPG if needed
            if source_file.endswith('.png'):
                try:
                    img = Image.open(source_file)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    img.save(target_path, 'JPEG', quality=95)
                    print(f"   ✅ Converted {source_file} to {target_file}")
                except Exception as e:
                    print(f"   ❌ Failed to convert {source_file}: {e}")
                    continue
            else:
                # Direct copy for JPG files
                try:
                    shutil.copy2(source_file, target_path)
                    print(f"   ✅ Copied {source_file} to {target_file}")
                except Exception as e:
                    print(f"   ❌ Failed to copy {source_file}: {e}")
                    continue

            setup_students.append((student_id, student_name, target_file))
        else:
            print(f"   ⚠️ {source_file} not found")

    return setup_students


def update_database(students):
    """Update database with your students"""
    if not students:
        print("❌ No students to add to database")
        return False

    print("\n🗄️ Updating database...")

    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()

        # Clear existing students
        cursor.execute("DELETE FROM students")
        cursor.execute("DELETE FROM trust_scores")
        cursor.execute("DELETE FROM attendance_events")

        # Add your students
        for student_id, student_name, filename in students:
            cursor.execute("""
                INSERT INTO students (id, name, avatar_url, seat_row, seat_col)
                VALUES (?, ?, ?, ?, ?)
            """, (student_id, student_name, f"/avatars/{filename}", 1, len(students)))

            cursor.execute("""
                INSERT INTO trust_scores (student_id, score, punctuality, consistency, streak)
                VALUES (?, 100, 100, 100, 0)
            """, (student_id,))

            print(f"   ✅ Added {student_name} (ID: {student_id})")

        # Add sample insights
        insights = [
            ("trend", "📈 Your attendance system is now active!", "high"),
            ("highlight", "🌟 Face recognition ready for Sai and Image Person", "high"),
            ("prediction", "📊 Ready to track attendance patterns", "med"),
        ]

        cursor.executemany("""
            INSERT INTO insights (kind, text, impact) VALUES (?, ?, ?)
        """, insights)

        conn.commit()
        conn.close()

        print("✅ Database updated successfully!")
        return True

    except Exception as e:
        print(f"❌ Database update failed: {e}")
        return False


def test_face_recognition(students):
    """Test face recognition on your images"""
    print("\n🔍 Testing face recognition...")

    try:
        import face_recognition

        for student_id, student_name, filename in students:
            filepath = os.path.join("known_faces", filename)
            try:
                image = face_recognition.load_image_file(filepath)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image)

                if face_locations:
                    print(f"   ✅ {student_name}: {len(face_locations)} face(s) detected")
                    if face_encodings:
                        print(f"      ✅ Face encoding generated successfully")
                else:
                    print(f"   ❌ {student_name}: No faces detected")
            except Exception as e:
                print(f"   ❌ {student_name}: Error - {e}")

        return True  # ✅ FIX: moved outside the loop

    except ImportError:
        print("   ⚠️ face_recognition not installed, skipping test")
        return True


def create_test_script(students):
    """Create test script for your specific students"""

    student_list = ", ".join([f'("{s[0]}", "{s[1]}")' for s in students])

    test_script = f'''#!/usr/bin/env python3
"""
Test Your Attendance System
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_server():
    """Test if server is running"""
    try:
        response = requests.get(f"{{BASE_URL}}/api/healthz")
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print(f"❌ Server error: {{response.status_code}}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {{e}}")
        print("💡 Make sure to run: python main.py")
        return False

def test_students():
    """Test students endpoint"""
    try:
        response = requests.get(f"{{BASE_URL}}/api/students")
        if response.status_code == 200:
            students = response.json()
            print(f"✅ Found {{len(students)}} students:")
            for student in students:
                print(f"   • {{student['name']}} ({{student['id']}}) - {{student.get('status', 'Absent')}}")
            return students
        else:
            print(f"❌ Students endpoint failed: {{response.status_code}}")
            return []
    except Exception as e:
        print(f"❌ Error: {{e}}")
        return []

def test_simulate_checkins():
    """Test simulate check-in for your students"""
    your_students = [{student_list}]

    print("\\n🎭 Testing simulate check-ins:")
    for student_id, student_name in your_students:
        data = {{"student_id": student_id, "status": "Present"}}
        try:
            response = requests.post(f"{{BASE_URL}}/api/simulate-checkin", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {{student_name}}: {{result['message']}}")
            else:
                print(f"   ❌ {{student_name}}: Failed ({{response.status_code}})")
        except Exception as e:
            print(f"   ❌ {{student_name}}: Error - {{e}}")

def main():
    print("🧪 TESTING YOUR ATTENDANCE SYSTEM")
    print("=" * 50)

    if not test_server():
        return

    print("\\n👥 Testing students...")
    test_students()

    print("\\n🎭 Testing check-ins...")
    test_simulate_checkins()

    print("\\n🌐 Next Steps:")
    print("1. Open Swagger UI: http://localhost:8000/docs")
    print("2. Try GET /api/students")
    print("3. Try POST /api/simulate-checkin")
    print("4. For real face recognition, use POST /api/checkin with image upload")

    print("\\n✅ Your backend is ready for the frontend team!")

if __name__ == "__main__":
    main()
'''

    # ✅ FIX: enforce UTF-8 encoding so emojis don't crash on Windows
    with open("test_your_system.py", "w", encoding="utf-8") as f:
        f.write(test_script)

    print(f"📜 Created test_your_system.py")


def main():
    """Main setup function"""
    print("🚀 SETTING UP YOUR ATTENDANCE SYSTEM IMAGES")
    print("=" * 60)

    # Setup images
    students = setup_your_images()

    if not students:
        print("\n❌ No images found or processed!")
        print("\n💡 Make sure you have:")
        print("   • sai.jpg (your Sai image)")
        print("   • image.png (your other person image)")
        print("in the same directory as this script")
        return False

    # Update database
    if not update_database(students):
        return False

    # Test face recognition
    test_face_recognition(students)

    # Create test script
    create_test_script(students)

    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)

    print(f"\n✅ Successfully set up {len(students)} students:")
    for student_id, student_name, filename in students:
        print(f"   • {student_name} (ID: {student_id}) → {filename}")

    print("\n📁 Files created:")
    for _, _, filename in students:
        print(f"   • known_faces/{filename}")

    print("\n🚀 Next Steps:")
    print("1. Start backend:")
    print("   python main.py")
    print()
    print("2. Test your system:")
    print("   python test_your_system.py")
    print()
    print("3. Open Swagger UI:")
    print("   http://localhost:8000/docs")
    print()
    print("4. Your API endpoints:")
    print("   • GET  /api/students")
    print("   • POST /api/simulate-checkin")
    print("   • POST /api/checkin (for real face recognition)")
    print()
    print("✅ Backend is ready for your frontend team!")

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
