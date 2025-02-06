import cv2 

# Load trained recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load Haar cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
 

# User IDs (ensure this matches your training IDs)
names = ['', 'vansh' ]  # Ensure you have the correct names in order

# Start video capture
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)  # Set width
cam.set(4, 480)  # Set height

# Set minimum face size for detection
minW = 0.1 * cam.get(3)
minH = 0.2 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Fix function name

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(
        converted_image, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face ROI (region of interest)
        face_roi = converted_image[y:y + h, x:x + w]

        if face_roi.size == 0:
            continue  # Skip if face_roi is empty

        id, confidence = recognizer.predict(face_roi)

        # Lower confidence means better match (range 0-100)
        if confidence < 50:  # Adjust the threshold if needed
            if id < len(names):
                name = names[id]
            else:
                name = "VANSH"
            confidence_text = "{0}%".format(round(100 - confidence))
        else:
            name = "Unknown"
            confidence_text = "{0}%".format(round(100 - confidence))

        # Display name & confidence on the frame
        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Face Recognition', img)

    # Exit when 'ESC' is pressed
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("Thanks for using the face recognition system!")
cam.release()
cv2.destroyAllWindows()
