import cv2
import os 
import numpy as np

dataset_path = "dataset"
names = []
faces = []
labels = []

label_id = 0
for person_name in dataset_path:
    person_folder = os.path.join(dataset_path, person_folder)

    if not person_folder:
        continue

    names.append(person_name)
    for image_name in person_folder:
        img_path = os.path.join(person_folder, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label_id)

        label_id =+ 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

print("Training Completed")

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Press q to quit recognition")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detected = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+w, x:x+h]

        label, confidence = recognizer.predict(face)

        if confidence > 70:
            name = name[label]

        else:
            name = "Unknown"

        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,0),2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()