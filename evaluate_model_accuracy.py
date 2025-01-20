import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Load the preprocessed data
with open('processed_data/preprocessed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test, person_to_label = pickle.load(f)

num_classes = len(np.unique(y_train))

# Train the model
model = models.Sequential([
    layers.Flatten(input_shape=(100, 100)),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)  # Adjust epochs as needed

# Save the trained model
model.save('model/face_recognition_model.h5')

# Compute training accuracy
y_train_pred = np.argmax(model.predict(X_train), axis=1)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Compute test accuracy
y_test_pred = np.argmax(model.predict(X_test), axis=1)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Save accuracies to a text file
with open('model/accuracy_results.txt', 'w') as f:
    f.write(f"Training Accuracy: {train_accuracy}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")