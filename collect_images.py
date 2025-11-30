import cv2
import os

if not os.path.exists("dataset"):
    os.makedirs("dataset")

name = input("Enter Person's Name")

person_dir = "dataset/" + name
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0 
print("Press 'c' to capture image, 'q' to quit")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == ord('c') and len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{person_dir}/{count}.jpg", face)
        print(f"Saved: {person_dir}/{count}.jpg")
        count =+ 1

    elif key == ord('q'):
        break

    elif key == ord('c') and len(faces) < 1:
        print("No faces Captured")

    else:
        print("Enter a valid Key")

cap.release()
cv2.destroyAllWindows()