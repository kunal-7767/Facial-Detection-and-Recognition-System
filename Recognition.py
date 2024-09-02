import cv2
import numpy as np

# Load the saved image and convert it to grayscale
kunal_image = cv2.imread("Kunal.jpg")
kunal_gray = cv2.cvtColor(kunal_image, cv2.COLOR_BGR2GRAY)

# Initialize the face recognizer
face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Detect the face in the saved image
kunal_faces = face_cascade.detectMultiScale(kunal_gray, 1.1, 4)
for (x, y, w, h) in kunal_faces:
    # Train the recognizer on the detected face
    recognizer.train([kunal_gray[y:y+h, x:x+w]], np.array([1]))

# Start the webcam to recognize your face in real-time
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = img_gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(face)
        
        # Check if the recognized face matches the saved face
        if id == 1 and confidence < 50:  
            name = "Kunal"
        else:
            name = "Unknown"
        
        # Display the name and rectangle around the face
        cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
