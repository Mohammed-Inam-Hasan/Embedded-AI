import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import serial

ser = serial.Serial('COM3', 115200, timeout=1)  # Change 'COM5' to the actual COM port of your STM32


# Load the trained face mask detection model
model = load_model("mask_detector.h5")

# Load OpenCV's pre-trained Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video stream
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(60, 60),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop over detected faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  # normalize

    (mask, withoutMask) = model.predict(face)[0]

    if mask > withoutMask:
            label = f"Mask: {mask*100:.2f}%"
            color = (0, 255, 0)  # Green
            ser.write(b'M')  # Send 'M' for Mask
    else:
            label = f"No Mask: {withoutMask*100:.2f}%"
            color = (0, 0, 255)  # Red
            ser.write(b'N')  # Send 'N' for No Mask


    # Display the label and bounding box rectangle
    cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show the output frame
    cv2.imshow("Face Mask Detector", frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
