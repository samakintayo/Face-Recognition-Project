import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = load_model('model/face_recognition_model.h5')

# Load the person_to_label mapping
with open('processed_data/preprocessed_data.pkl', 'rb') as f:
    _, X_test, _, _, person_to_label = pickle.load(f)

# Test the model on the testing data
for i in range(len(X_test)):
    input_frame = np.expand_dims(X_test[i], axis=0)
    predictions = model.predict(input_frame)
    predicted_label = np.argmax(predictions)

    # Display the person's name on the frame
    person_name = [name for name, label in person_to_label.items() if label == predicted_label][0]
    cv2.putText(X_test[i], person_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255)),