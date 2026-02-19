import cv2
import mediapipe as mp
import numpy as np
import pygame
import time



pygame.mixer.init()

try:
    pygame.mixer.music.load("alarm.mp3") 
except:
    print("Warning: alarm.mp3 not found.")

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


eye_closed_start_time = 0
is_eye_closed = False
max_time = 5.0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
        
            for eye_indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
                coords = np.array([(int(face_landmarks.landmark[i].x * w), 
                                    int(face_landmarks.landmark[i].y * h)) for i in eye_indices])
                ex, ey, ew, eh = cv2.boundingRect(coords)
                cv2.rectangle(image, (ex-10, ey-10), (ex+ew+10, ey+eh+10), (0, 0, 255), 2)

            left_upper = face_landmarks.landmark[159].y
            left_lower = face_landmarks.landmark[145].y
            dist = left_lower - left_upper

            if dist < 0.015:

                if not is_eye_closed:
                    eye_closed_start_time = time.time()
                    is_eye_closed = True

                elapsed_time = time.time() - eye_closed_start_time

                if elapsed_time >= max_time:
                    cv2.putText(image, "SLEEPING!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)
                else:
                    cv2.putText(image, f"Closing... {elapsed_time:.1f}s", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
            else:
                is_eye_closed = False
                eye_closed_start_time = 0
                cv2.putText(image, "AWAKE", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                pygame.mixer.music.stop()

    cv2.imshow('Sleep Detector (2s Delay)', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()