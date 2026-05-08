import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import webbrowser
import time
import os
import urllib.request
import numpy as np

MEDIA_PATH = r"C:\Users\PC\Documents\GitHub\ora-bwakal-face\ora-bwakal-face.mp4"

# Sensitivitas (Semakin kecil nilainya, semakin sensitif terhadap bukaan mulut)
MOUTH_THRESHOLD = 0.06 
# Jeda waktu agar media tidak terbuka berkali-kali secara brutal
COOLDOWN_SECONDS = 5 
# ---------------------------

# Download model file jika belum ada
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Mengunduh model face landmarker...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model berhasil diunduh!")

# Konfigurasi Face Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_trigger_time = 0

print("Sistem berjalan... Buka mulut atau julurkan lidah untuk memicu media.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip gambar agar seperti cermin
    image = cv2.flip(image, 1)
    
    # Convert to MediaPipe Image format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Detect face landmarks
    detection_result = detector.detect(mp_image)
    
    current_time = time.time()

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Ambil koordinat Y untuk bibir atas (13) dan bibir bawah (14)
            upper_lip = face_landmarks[13].y
            lower_lip = face_landmarks[14].y

            # Hitung jarak bukaan mulut
            mouth_distance = lower_lip - upper_lip

            # Logika Pemicu
            if mouth_distance > MOUTH_THRESHOLD:
                if (current_time - last_trigger_time) > COOLDOWN_SECONDS:
                    print(f"Gestur Terdeteksi! Membuka: {MEDIA_PATH}")
                    
                    # Membuka media (Bisa file lokal atau URL)
                    webbrowser.open(MEDIA_PATH)
                    
                    last_trigger_time = current_time
                
                # Beri indikasi visual di layar
                cv2.putText(image, "ORA BWAKAL FACE!", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Gambar landmark di wajah
            h, w, _ = image.shape
            for landmark in face_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 1, (200, 200, 200), -1)

    # Tampilkan Preview Kamera
    cv2.putText(image, "Press 'ESC' to exit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow('Face Gesture Media Controller', image)

    if cv2.waitKey(5) & 0xFF == 27: # Keluar dengan tombol ESC
        break

cap.release()
cv2.destroyAllWindows()