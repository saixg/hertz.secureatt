import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import json

class MLService:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from the known_faces directory"""
        known_faces_dir = "known_faces"
        
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created {known_faces_dir} directory. Please add student photos here.")
            return
        
        self.known_encodings = []
        self.known_names = []
        
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(known_faces_dir, filename)
                try:
                    img = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        encoding = encodings[0]  # Take first face
                        self.known_encodings.append(encoding)
                        name = os.path.splitext(filename)[0]
                        self.known_names.append(name)
                        print(f"Loaded face for: {name}")
                    else:
                        print(f"No face found in {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.known_encodings)} known faces")
    
    def decode_base64_image(self, base64_string):
        """Decode base64 image string to numpy array"""
        try:
            # Strip prefix if present (like data:image/png;base64,)
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            pil_image = Image.open(BytesIO(image_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            frame = np.array(pil_image)
            return frame
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def simple_liveness_check(self, frame):
        """Basic liveness check (replace with real anti-spoofing)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        contrast = gray.std()
        brightness = gray.mean()
        if contrast < 20:
            return False, "Low contrast detected"
        if brightness < 30 or brightness > 220:
            return False, "Poor lighting conditions"
        return True, "Live face detected"
    
    def process_checkin(self, base64_image):
        try:
            frame = self.decode_base64_image(base64_image)
            if frame is None:
                return {
                    "student_id": None,
                    "student_name": None,
                    "status": "Suspicious",
                    "confidence": 0.0,
                    "message": "Could not decode image"
                }
            
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            if not face_locations:
                return {
                    "student_id": None,
                    "student_name": None,
                    "status": "Unknown",
                    "confidence": 0.0,
                    "message": "No face detected"
                }
            
            face_encoding = face_encodings[0]
            name = "Unknown"
            student_id = None
            confidence = 0.0
            
            if self.known_encodings:
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        confidence = 1.0 - face_distances[best_match_index]
                        
                        name_to_id = {
                            "hasini": 1,
                            "anji": 2,
                            "venkat": 3,
                            "sai_reddy": 4,
                            "sai": 4
                        }
                        student_id = name_to_id.get(name.lower())
            
            is_live, liveness_message = self.simple_liveness_check(frame)
            
            if name == "Unknown":
                status = "Unknown"
                message = "Unknown person detected"
            elif not is_live:
                status = "Suspicious"
                message = f"Potential spoofing detected: {liveness_message}"
                confidence = 0.0
            else:
                current_time = datetime.now().time()
                if current_time.hour >= 9 and current_time.minute > 5:
                    status = "Late"
                    message = f"Late arrival detected for {name}"
                else:
                    status = "Present"
                    message = f"On time arrival for {name}"
            
            return {
                "student_id": student_id,
                "student_name": name if name != "Unknown" else None,
                "status": status,
                "confidence": confidence,
                "message": message
            }
        
        except Exception as e:
            print(f"Error processing check-in: {e}")
            return {
                "student_id": None,
                "student_name": None,
                "status": "Suspicious",
                "confidence": 0.0,
                "message": f"Processing error: {str(e)}"
            }

# Global instance of your ML service
ml_service = MLService()

def process_checkin_frame(base64_image):
    """Function your friend can call"""
    return ml_service.process_checkin(base64_image)

