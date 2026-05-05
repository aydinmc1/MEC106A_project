import cv2
import numpy as np

# Initialize webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load cascade classifiers for different object detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
# Car cascade not available in default OpenCV
# car_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_cars.xml'
# )

# Detection parameters
detect_faces = True
detect_eyes = True
detect_cars = False  # Car detection not available

print("Starting camera stream...")
print("Press 'f' to toggle face detection")
print("Press 'e' to toggle eye detection")
print("Press 'c' to toggle car detection")
print("Press 'q' to quit")
print()

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from camera")
        break
    
    # Convert frame to grayscale for detection (cascade classifiers work on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    if detect_faces:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # Draw rectangle around face (green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add label
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Detect eyes within face region
            if detect_eyes:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    # Draw rectangle around eyes (blue)
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
    
    # Detect cars (disabled - cascade not available)
    # if detect_cars:
    #     cars = car_cascade.detectMultiScale(gray, 1.1, 4)
    #     for (x, y, w, h) in cars:
    #         # Draw rectangle around car (red)
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         # Add label
    #         cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Add status text at top
    status = f"Faces: {'ON' if detect_faces else 'OFF'} | Eyes: {'ON' if detect_eyes else 'OFF'} | Cars: {'ON' if detect_cars else 'OFF'}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the frame with detections
    cv2.imshow('Object Detection - Press q to quit', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('f'):
        detect_faces = not detect_faces
        print(f"Face detection: {'ON' if detect_faces else 'OFF'}")
    elif key == ord('e'):
        detect_eyes = not detect_eyes
        print(f"Eye detection: {'ON' if detect_eyes else 'OFF'}")
    elif key == ord('c'):
        detect_cars = not detect_cars
        print(f"Car detection: {'ON' if detect_cars else 'OFF'}")

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Camera released. Done!")
