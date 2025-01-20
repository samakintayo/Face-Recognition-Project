import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import time  # Import the time module

# Load the pre-trained face recognition model
model = load_model('model/face_recognition_model.h5')

# Load the person-to-label mapping
with open('processed_data/preprocessed_data.pkl', 'rb') as f:
    _, _, _, _, person_to_label = pickle.load(f)

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Perform face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face for model input
        face_roi_resized = cv2.resize(face_roi, (100, 100))

        # Normalize pixel values
        face_roi_normalized = face_roi_resized / 255.0

        # Expand dimensions for model input
        face_input = np.expand_dims(face_roi_normalized, axis=0)

        # Predict the label for the face
        predicted_label = np.argmax(model.predict(face_input), axis=1)[0]

        # Map the label to the person's name
        person_name = [name for name, label in person_to_label.items() if label == predicted_label][0]

        # Draw a rectangle around the face and display the person's name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Face Recognition - Live Feed', frame)

    # Add a delay to control frame capture speed (e.g., 0.5 seconds)
    time.sleep(0.0)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
