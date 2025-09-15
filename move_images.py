#!/usr/bin/env python3
"""
Move Your Images to Known Faces Directory
This script helps you move your existing images to the right place
"""

import os
import shutil
import face_recognition
from PIL import Image

def find_your_images():
    """Find your images in the current directory"""
    print("🔍 Looking for your images...")
    
    # Look for your images in current directory and common subdirectories
    search_locations = [
        ".",  # Current directory
        "images/",
        "photos/",
        "../",  # Parent directory
    ]
    
    target_files = ["sai.jpg", "image.png", "me.jpg", "hasini.jpg", "image.jpg"]
    found_images = {}
    
    for location in search_locations:
        if os.path.exists(location):
            files = os.listdir(location)
            for target in target_files:
                if target in files:
                    full_path = os.path.join(location, target)
                    found_images[target] = full_path
                    print(f"   ✅ Found: {target} at {full_path}")
    
    if not found_images:
        print("   ❌ No target images found in common locations")
        print("   📁 Looking for any image files...")
        
        # Look for any image files
        for location in search_locations:
            if os.path.exists(location):
                files = os.listdir(location)
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        full_path = os.path.join(location, file)
                        found_images[file] = full_path
                        print(f"   📸 Found image: {file} at {full_path}")
    
    return found_images

def copy_and_rename_images(found_images):
    """Copy and rename images to known_faces directory"""
    print("\n📁 Setting up known_faces directory...")
    
    # Ensure known_faces exists
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    
    # Mapping rules for your images
    rename_mapping = {
        "sai.jpg": "sai.jpg",
        "image.png": "image_person.jpg", 
        "image.jpg": "image_person.jpg",
        "me.jpg": "me.jpg",
        "hasini.jpg": "hasini.jpg"
    }
    
    copied_files = []
    
    for original_name, source_path in found_images.items():
        # Determine target name
        if original_name in rename_mapping:
            target_name = rename_mapping[original_name]
        else:
            # For any other image, use the original name but ensure .jpg extension
            base_name = os.path.splitext(original_name)[0]
            target_name = f"{base_name}.jpg"
        
        target_path = os.path.join(known_faces_dir, target_name)
        
        try:
            # Copy and convert if needed
            if original_name.lower().endswith('.png'):
                # Convert PNG to JPG
                img = Image.open(source_path)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(target_path, 'JPEG', quality=95)
                print(f"   ✅ Converted {original_name} → {target_name}")
            else:
                # Direct copy for JPG files
                shutil.copy2(source_path, target_path)
                print(f"   ✅ Copied {original_name} → {target_name}")
            
            copied_files.append(target_name)
            
        except Exception as e:
            print(f"   ❌ Failed to copy {original_name}: {e}")
    
    return copied_files

def test_face_recognition(image_files):
    """Test face recognition on the copied files"""
    print("\n🔍 Testing face recognition on your images...")
    
    valid_faces = []
    
    for filename in image_files:
        filepath = os.path.join("known_faces", filename)
        
        try:
            print(f"\n   📸 Testing {filename}...")
            
            # Load image
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_locations) == 0:
                print(f"      ❌ No faces detected")
                print(f"      💡 Make sure the image is clear and shows a face")
            elif len(face_locations) > 1:
                print(f"      ⚠️ Multiple faces detected ({len(face_locations)})")
                print(f"      💡 Will use the first face found")
                valid_faces.append(filename)
            else:
                print(f"      ✅ Perfect! 1 face detected")
                valid_faces.append(filename)
                
            # Test encoding generation
            if face_encodings:
                encoding_size = len(face_encodings[0])
                print(f"      ✅ Face encoding generated ({encoding_size} features)")
            
        except Exception as e:
            print(f"      ❌ Error processing {filename}: {e}")
    
    return valid_faces

def update_database_for_found_images(valid_faces):
    """Update database based on which images we actually have"""
    print(f"\n🗄️ Updating database for {len(valid_faces)} valid images...")
    
    try:
        import sqlite3
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Clear existing students
        cursor.execute("DELETE FROM students")
        cursor.execute("DELETE FROM trust_scores")
        
        student_mappings = {
            "sai.jpg": ("sai", "Sai"),
            "image_person.jpg": ("image_person", "Image Person"),
            "me.jpg": ("me", "Me"),
            "hasini.jpg": ("hasini", "Hasini"),
        }
        
        added_students = []
        
        for filename in valid_faces:
            if filename in student_mappings:
                student_id, student_name = student_mappings[filename]
                
                # Add student
                cursor.execute("""
                    INSERT INTO students (id, name, avatar_url, seat_row, seat_col)
                    VALUES (?, ?, ?, ?, ?)
                """, (student_id, student_name, f"/avatars/{filename}", 1, len(added_students) + 1))
                
                # Add trust score
                cursor.execute("""
                    INSERT INTO trust_scores (student_id, score, punctuality, consistency, streak)
                    VALUES (?, 100, 100, 100, 0)
                """, (student_id,))
                
                added_students.append((student_id, student_name))
                print(f"   ✅ Added to database: {student_name} (ID: {student_id})")
        
        conn.commit()
        conn.close()
        
        return added_students
        
    except Exception as e:
        print(f"   ❌ Database update failed: {e}")
        return []

def create_test_script_for_your_images(students):
    """Create a custom test script for your specific images"""
    
    test_content = f'''#!/usr/bin/env python3
"""
Test Your Specific Students
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_your_students():
    """Test your specific students"""
    print("👥 Your Students:")
    
    try:
        response = requests.get(f"{{BASE_URL}}/api/students")
        if response.status_code == 200:
            students = response.json()
            for student in students:
                print(f"   • {{student['name']}} (ID: {{student['id']}}) - {{student.get('status', 'Absent')}}")
        else:
            print(f"❌ Failed to get students: {{response.status_code}}")
    except Exception as e:
        print(f"❌ Error: {{e}}")

def test_checkin():
    """Test check-in for each of your students"""
    students = {repr([(s[0], s[1]) for s in students])}
    
    print("\\n🎭 Testing check-ins:")
    for student_id, student_name in students:
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
    print("🧪 Testing Your Attendance System")
    print("=" * 40)
    
    test_your_students()
    test_checkin()
    
    print("\\n🌐 Next steps:")
    print("1. Open Swagger UI: http://localhost:8000/docs")
    print("2. Try the endpoints with your student IDs")
    print("3. For real face recognition, use POST /api/checkin with image files")

if __name__ == "__main__":
    main()
'''
    
    with open("test_my_students.py", "w") as f:
        f.write(test_content)
    
    print("📜 Created test_my_students.py for your specific setup")

def main():
    """Main function to move and set up images"""
    print("📸 MOVING YOUR IMAGES TO ATTENDANCE SYSTEM")
    print("=" * 50)
    
    # Find your images
    found_images = find_your_images()
    
    if not found_images:
        print("\n❌ No images found!")
        print("\n💡 Please make sure you have image files in:")
        print("   • Current directory")
        print("   • Images with faces (sai.jpg, image.png, etc.)")
        print("\n📋 Manual steps:")
        print("1. Copy your image files to the 'known_faces/' directory")
        print("2. Rename them to match student IDs (e.g., sai.jpg)")
        print("3. Run: python main.py")
        return False
    
    print(f"\n✅ Found {len(found_images)} images")
    
    # Copy and rename images
    copied_files = copy_and_rename_images(found_images)
    
    if not copied_files:
        print("\n❌ No images were copied successfully")
        return False
    
    print(f"\n✅ Successfully copied {len(copied_files)} images")
    
    # Test face recognition
    valid_faces = test_face_recognition(copied_files)
    
    if not valid_faces:
        print("\n❌ No valid faces detected in any image")
        print("💡 Make sure your images:")
        print("   • Are clear and well-lit")
        print("   • Show faces directly")
        print("   • Are not too blurry or dark")
        return False
    
    print(f"\n✅ {len(valid_faces)} images have valid faces")
    
    # Update database
    students = update_database_for_found_images(valid_faces)
    
    if not students:
        print("\n❌ Failed to update database")
        return False
    
    # Create test script
    create_test_script_for_your_images(students)
    
    print("\n" + "=" * 50)
    print("🎉 IMAGES SETUP COMPLETE!")
    print("=" * 50)
    
    print(f"\n✅ Successfully set up {len(students)} students:")
    for student_id, student_name in students:
        print(f"   • {student_name} (ID: {student_id})")
    
    print(f"\n📁 Images in known_faces/:")
    for filename in valid_faces:
        print(f"   • {filename}")
    
    print("\n🚀 Next steps:")
    print("1. Start backend: python main.py")
    print("2. Test system: python test_my_students.py") 
    print("3. Open Swagger: http://localhost:8000/docs")
    print("4. Your backend is ready for frontend!")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)