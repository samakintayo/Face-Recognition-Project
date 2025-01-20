import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import shutil
import os

def preprocess_and_save(image_folder, save_folder):
    images = []
    labels = []

    person_to_label = {}  # Mapping from person names to labels
    label_count = 0

    for person_folder in os.listdir(image_folder):
        person_path = os.path.join(image_folder, person_folder)
        person_name = person_folder.split('_')[0]

        person_to_label[person_name] = label_count

        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            image = cv2.resize(image, (100, 100))  # Resize for consistency

            images.append(image)
            labels.append(label_count)

        # Move raw data to processed folder
        shutil.move(person_path, os.path.join(save_folder, person_folder))

        label_count += 1

    images = np.array(images)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Save the preprocessed data
    with open(os.path.join(save_folder, 'preprocessed_data.pkl'), 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test, person_to_label), f)

    return X_train, X_test, y_train, y_test, person_to_label

# Example usage
image_folder = "dataset"
save_folder = "processed_data"
X_train, X_test, y_train, y_test, person_to_label = preprocess_and_save(image_folder, save_folder)