import cv2
import os


# Set the base path for the dataset
base_path = "dataset"

# Create the base directory if it doesn't exist
os.makedirs(base_path, exist_ok=True)

# Function to capture and save images for a specific label and person
def capture_images(label, person, num_samples=100):
    # Set the path for the specific label and person
    person_path = os.path.join(base_path, label, person)
    os.makedirs(person_path, exist_ok=True)

    print(f"Capturing {num_samples} images for label {label} and person {person}")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # Change to 1 or 2 if you have multiple cameras

    # Face cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Counter for the number of captured images
    count = 0

    while count < num_samples:
        # Capture frame from webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the face region
            face_roi = gray[y:y + h, x:x + w]

            # Save the face image to the corresponding directory
            cv2.imwrite(os.path.join(person_path, f"{person}_{count}.png"), face_roi)

            count += 1

        # Display the frame
        cv2.imshow('Capture Faces', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Capture images for different labels (e.g., "positive") and people
capture_images(person="Person_label", num_samples=100)
# capture_images(label="positive", person="person2", num_samples=100)
# Add more calls for different people as needed