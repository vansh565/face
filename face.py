from flask import Flask, jsonify
import cv2
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["recognized_faces"]

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

names = {1: "Vansh", 2: "Alice", 3: "Bob"}

cam = cv2.VideoCapture(0)

@app.route("/recognize", methods=["GET"])
def recognize():
    ret, img = cam.read()
    if not ret:
        return jsonify({"error": "Failed to capture image"}), 500

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        confidence_text = round(100 - confidence)

        name = names.get(id, "Unknown")

        # Store in MongoDB
        collection.insert_one({
            "id": id,
            "name": name,
            "confidence": confidence_text,
            "timestamp": datetime.now()
        })

        return jsonify({"id": id, "name": name, "confidence": confidence_text})

    return jsonify({"name": "No face detected"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
