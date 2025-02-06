import cv2
import numpy as np
from PIL import Image
import os

# Path to face samples
path = 'samples'

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def Images_And_Labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) ]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        gray_img = Image.open(imagePath).convert('L')  # Convert image to grayscale
        img_arr = np.array(gray_img, 'uint8')  # Convert to NumPy array

        id = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract user ID from filename
        faces = detector.detectMultiScale(img_arr, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

print("Training faces... Please wait.")

faces, ids = Images_And_Labels(path)

# Train the model
recognizer.train(faces, np.array(ids))

# Ensure trainer directory exists
if not os.path.exists("trainer"):
    os.makedirs("trainer")

# Save the trained model
recognizer.save('trainer/trainer.yml')

print("Model trained successfully! You can now recognize faces.")
