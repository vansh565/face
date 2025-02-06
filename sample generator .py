
import cv2
import os

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set width
cam.set(4, 480)  # Set height

# Load the face detection classifier
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the samples folder if it doesn't exist
if not os.path.exists("samples"):
    os.makedirs("samples")

face_id = input("Enter a Numeric user ID here: ")
name= input("enter user name")
print("Taking samples, look at the camera...")

count = 0


while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Corrected function name
    faces = detector.detectMultiScale(converted_image, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save face samples
        cv2.imwrite(f"samples/face.{face_id}.{count}.jpg", converted_image[y:y + h, x:x + w])

    cv2.imshow('Capturing Face Samples', img)  # Corrected function name

    k = cv2.waitKey(100) & 0xFF
    if k == 27:  # Press 'ESC' to exit
        break
    elif count >= 10:  # Collect 10 samples
        break

print("Samples taken. Closing the program...")
cam.release()
cv2.destroyAllWindows()
