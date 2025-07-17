import cv2
import numpy as np

def detect_faces_webcam():
    # Load the pre-trained face and smile detection models
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Start video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            
            # Detect smile
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
            
            # Estimate emotion based on smile detection
            if len(smiles) > 0:
                text = "Smiling (Happy)"
                color = (0, 255, 0)
            else:
                # Simple heuristic: if no smile and face detected, assume neutral or sad
                emotion_intensity = np.random.rand()  # Simulating intensity detection
                if emotion_intensity < 0.3:
                    text = "Sad"
                    color = (255, 0, 255)
                elif emotion_intensity < 0.6:
                    text = "Neutral"
                    color = (0, 255, 255)
                else:
                    text = "Angry"
                    color = (0, 0, 255)
            
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display the output
        cv2.imshow('Face and Emotion Detection', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
detect_faces_webcam()
