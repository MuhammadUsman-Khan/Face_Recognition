import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import time
import json

# Configuration
CONFIG = {
    "dataset_path": "dataset",
    "min_confidence": 70,
    "scale_factor": 1.3,
    "min_neighbors": 5,
    "min_face_size": (30, 30),
    "max_faces_per_person": 50
}

# Global variables
cap = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0
person_dir = ""
recognizer = None
names = []
faces = []
labels = []
label_id = 0
recognition_running = False
training_data_loaded = False
current_mode = "normal"  # normal, capture, recognition

# Initialize directories
if not os.path.exists(CONFIG["dataset_path"]):
    os.makedirs(CONFIG["dataset_path"])

def load_training_data():
    """Load existing training data if available"""
    global names, faces, labels, label_id, training_data_loaded
    
    names_file = os.path.join(CONFIG["dataset_path"], "names.json")
    if os.path.exists(names_file):
        try:
            with open(names_file, 'r') as f:
                names = json.load(f)
            training_data_loaded = True
            update_status(f"Loaded {len(names)} persons from training data")
        except Exception as e:
            print(f"Error loading training data: {e}")

def save_training_data():
    """Save training data for persistence"""
    try:
        names_file = os.path.join(CONFIG["dataset_path"], "names.json")
        with open(names_file, 'w') as f:
            json.dump(names, f)
    except Exception as e:
        print(f"Error saving training data: {e}")

def get_name_and_create_folder():
    global person_dir, count, current_mode
    name = simpledialog.askstring("Input", "Enter Person's Name:")
    if name:
        person_dir = f"{CONFIG['dataset_path']}/{name}"
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        count = 0
        current_mode = "capture"
        update_status(f"Ready to capture faces for: {name}")
        messagebox.showinfo("Info", f"Folder created for {name}. Click 'Capture Face' to start capturing.")
    else:
        messagebox.showwarning("Warning", "Name cannot be empty!")

def start_camera():
    global cap, current_mode
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera!")
            return
    
    current_mode = "normal"
    update_status("Camera started - Normal mode")
    update_frame()

def update_frame():
    if cap and not recognition_running and current_mode != "recognition":
        ret, frame = cap.read()
        if ret:
            # Add timestamp to frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if current_mode == "capture":
                # Show face detection boxes in capture mode
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_detected = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=CONFIG["scale_factor"],
                    minNeighbors=CONFIG["min_neighbors"],
                    minSize=CONFIG["min_face_size"]
                )
                
                for (x, y, w, h) in faces_detected:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            display_frame(frame)
        
        root.after(10, update_frame)

def display_frame(frame, recog_names=None, recog_faces=None):
    # Resize frame for better display
    height, width = frame.shape[:2]
    max_width = 800
    max_height = 600
    
    if width > max_width or height > max_height:
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

def stop_camera():
    global cap, recognition_running, current_mode
    recognition_running = False
    current_mode = "normal"
    if cap:
        cap.release()
        cap = None
    video_label.configure(image="")
    update_status("Camera stopped")

def capture_face():
    global count
    if cap is None:
        messagebox.showwarning("Warning", "Camera not started!")
        return
    
    if not person_dir:
        messagebox.showwarning("Warning", "Please create a person folder first!")
        return

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=CONFIG["scale_factor"],
        minNeighbors=CONFIG["min_neighbors"],
        minSize=CONFIG["min_face_size"]
    )

    if len(faces_detected) > 0:
        x, y, w, h = faces_detected[0]
        face = gray[y:y+h, x:x+w]
        
        # Enhance face image
        face = cv2.equalizeHist(face)
        
        filename = f"{person_dir}/{count}.jpg"
        cv2.imwrite(filename, face)
        count += 1
        
        remaining = CONFIG["max_faces_per_person"] - count
        update_status(f"Saved: {filename} ({count}/{CONFIG['max_faces_per_person']}) - {remaining} remaining")
        
        if count >= CONFIG["max_faces_per_person"]:
            messagebox.showinfo("Info", f"Maximum {CONFIG['max_faces_per_person']} faces captured for this person.")
            current_mode = "normal"
    else:
        messagebox.showwarning("Warning", "No face detected! Ensure good lighting and face visibility.")

def train_recognizer():
    global faces, labels, names, label_id, recognizer
    
    # Check if dataset exists and has data
    if not os.path.exists(CONFIG["dataset_path"]) or not os.listdir(CONFIG["dataset_path"]):
        messagebox.showwarning("Warning", "No dataset found! Please capture faces first.")
        return
    
    progress_window = tk.Toplevel(root)
    progress_window.title("Training Progress")
    progress_window.geometry("300x100")
    progress_window.transient(root)
    progress_window.grab_set()
    
    progress_label = tk.Label(progress_window, text="Training in progress...")
    progress_label.pack(pady=10)
    
    progress = ttk.Progressbar(progress_window, mode='indeterminate')
    progress.pack(pady=10)
    progress.start()
    
    def training_task():
        global faces, labels, names, label_id, recognizer
        
        names = []
        faces = []
        labels = []
        label_id = 0
        
        for person_name in os.listdir(CONFIG["dataset_path"]):
            person_folder = os.path.join(CONFIG["dataset_path"], person_name)
            if os.path.isdir(person_folder) and os.listdir(person_folder):
                names.append(person_name)
                for image_name in os.listdir(person_folder):
                    if image_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(person_folder, image_name)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Resize images to consistent size for better recognition
                            img = cv2.resize(img, (200, 200))
                            faces.append(img)
                            labels.append(label_id)
                label_id += 1
        
        if len(faces) == 0:
            progress_window.destroy()
            messagebox.showwarning("Warning", "No valid face images found for training!")
            return
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        
        save_training_data()
        
        progress_window.destroy()
        messagebox.showinfo("Info", f"Training Completed! Trained on {len(faces)} images of {len(names)} persons.")
        update_status(f"Trained on {len(faces)} images of {len(names)} persons")
    
    # Run training in main thread to avoid GUI issues
    root.after(100, training_task)

def start_recognition():
    global recognition_running, current_mode
    if recognizer is None:
        messagebox.showwarning("Warning", "Train the recognizer first!")
        return
    
    if cap is None:
        start_camera()
    
    recognition_running = True
    current_mode = "recognition"
    update_status("Recognition mode active")
    recognize_frame()

def recognize_frame():
    global recognition_running
    if cap and recognition_running:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=CONFIG["scale_factor"],
                minNeighbors=CONFIG["min_neighbors"],
                minSize=CONFIG["min_face_size"]
            )

            for (x, y, w, h) in faces_detected:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                
                label, confidence = recognizer.predict(face_roi)
                
                if confidence < CONFIG["min_confidence"] and label < len(names):
                    name = names[label]
                    color = (0, 255, 0)  # Green for recognized
                    confidence_text = f"{name} ({100-confidence:.1f}%)"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    confidence_text = "Unknown"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, confidence_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Recognition Mode - {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            display_frame(frame)

        if recognition_running:
            root.after(10, recognize_frame)

def stop_recognition():
    global recognition_running, current_mode
    recognition_running = False
    current_mode = "normal"
    update_status("Recognition stopped")
    if cap:
        update_frame()

def update_status(message):
    status_bar.config(text=f"Status: {message}")

def show_dataset_info():
    """Show information about the current dataset"""
    if not os.path.exists(CONFIG["dataset_path"]):
        messagebox.showinfo("Dataset Info", "No dataset found!")
        return
    
    persons = []
    total_images = 0
    
    for person_name in os.listdir(CONFIG["dataset_path"]):
        person_folder = os.path.join(CONFIG["dataset_path"], person_name)
        if os.path.isdir(person_folder):
            image_count = len([f for f in os.listdir(person_folder) 
                             if f.endswith(('.jpg', '.png', '.jpeg'))])
            persons.append((person_name, image_count))
            total_images += image_count
    
    info_text = f"Dataset Information:\n\nTotal Persons: {len(persons)}\nTotal Images: {total_images}\n\n"
    
    for person, count in persons:
        info_text += f"{person}: {count} images\n"
    
    messagebox.showinfo("Dataset Info", info_text)

def clear_dataset():
    """Clear all dataset data"""
    if messagebox.askyesno("Confirm", "This will delete ALL dataset files. Continue?"):
        try:
            for person_name in os.listdir(CONFIG["dataset_path"]):
                person_folder = os.path.join(CONFIG["dataset_path"], person_name)
                if os.path.isdir(person_folder):
                    for file in os.listdir(person_folder):
                        os.remove(os.path.join(person_folder, file))
                    os.rmdir(person_folder)
            
            # Remove training data file
            names_file = os.path.join(CONFIG["dataset_path"], "names.json")
            if os.path.exists(names_file):
                os.remove(names_file)
            
            update_status("Dataset cleared")
            messagebox.showinfo("Info", "Dataset cleared successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error clearing dataset: {e}")

# Create main window
root = tk.Tk()
root.title("Advanced Face Recognition System")
root.geometry("1000x800")
root.configure(bg="#2C3E50")

# Load existing training data
load_training_data()

# Create GUI
title_label = tk.Label(root, text="Advanced Face Recognition System", 
                      font=("Arial", 24, "bold"), bg="#2C3E50", fg="white")
title_label.pack(pady=20)

video_label = tk.Label(root, bg="black", relief="sunken")
video_label.pack(pady=10, padx=20, fill="both", expand=True)

# Main button frame
button_frame = tk.Frame(root, bg="#2C3E50")
button_frame.pack(pady=20)

# Row 1
btn_name = tk.Button(button_frame, text="Create Person Folder", 
                    command=get_name_and_create_folder, width=18, bg="#1ABC9C", fg="white")
btn_name.grid(row=0, column=0, padx=5, pady=5)

btn_start = tk.Button(button_frame, text="Start Camera", 
                     command=start_camera, width=18, bg="#3498DB", fg="white")
btn_start.grid(row=0, column=1, padx=5, pady=5)

btn_capture = tk.Button(button_frame, text="Capture Face", 
                       command=capture_face, width=18, bg="#E67E22", fg="white")
btn_capture.grid(row=0, column=2, padx=5, pady=5)

# Row 2
btn_train = tk.Button(button_frame, text="Train Recognizer", 
                     command=train_recognizer, width=18, bg="#9B59B6", fg="white")
btn_train.grid(row=1, column=0, padx=5, pady=5)

btn_recognize = tk.Button(button_frame, text="Start Recognition", 
                         command=start_recognition, width=18, bg="#E74C3C", fg="white")
btn_recognize.grid(row=1, column=1, padx=5, pady=5)

btn_stop = tk.Button(button_frame, text="Stop Recognition", 
                    command=stop_recognition, width=18, bg="#34495E", fg="white")
btn_stop.grid(row=1, column=2, padx=5, pady=5)

# Row 3 - Utility buttons
btn_info = tk.Button(button_frame, text="Dataset Info", 
                    command=show_dataset_info, width=18, bg="#16A085", fg="white")
btn_info.grid(row=2, column=0, padx=5, pady=5)

btn_clear = tk.Button(button_frame, text="Clear Dataset", 
                     command=clear_dataset, width=18, bg="#C0392B", fg="white")
btn_clear.grid(row=2, column=1, padx=5, pady=5)

btn_stop_camera = tk.Button(button_frame, text="Stop Camera", 
                           command=stop_camera, width=18, bg="#34495E", fg="white")
btn_stop_camera.grid(row=2, column=2, padx=5, pady=5)

# Status bar
status_bar = tk.Label(root, text="Status: Ready", bd=1, relief="sunken", anchor="w",
                     bg="#34495E", fg="white", font=("Arial", 10))
status_bar.pack(side="bottom", fill="x")

# Instructions
instructions = tk.Label(root, 
                       text="Instructions: 1) Create Folder → 2) Start Camera → 3) Capture Faces → 4) Train → 5) Recognize",
                       bg="#2C3E50", fg="#BDC3C7", font=("Arial", 10))
instructions.pack(side="bottom", pady=5)

root.mainloop()