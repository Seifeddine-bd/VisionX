import tkinter as tk
# Import required libraries
from keras.models import load_model
import numpy as np
from tkinter import filedialog, messagebox
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class EnhancedFaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionX")
        self.root.geometry("800x600")
        icon_path = os.path.join(os.path.dirname(__file__), "visionX.ico")
        self.root.iconbitmap(icon_path)
        # Fixed image display area
        self.image_frame = tk.Frame(self.root, width=500, height=500, bg="gray")
        self.image_frame.pack(side=tk.TOP, pady=20)
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        # Buttons for features
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, pady=20)

        load_button = tk.Button(btn_frame, text="Load Image", command=self.load_image, width=15)
        load_button.grid(row=0, column=0, padx=5, pady=5)

        camera_button = tk.Button(btn_frame, text="Open Camera", command=self.open_camera, width=15)
        camera_button.grid(row=0, column=1, padx=5, pady=5)

        grayscale_button = tk.Button(btn_frame, text="Convert to Grayscale", command=self.convert_to_grayscale, width=15)
        grayscale_button.grid(row=0, column=2, padx=5, pady=5)

        face_detection_button = tk.Button(btn_frame, text="Detect Faces", command=self.detect_faces, width=15)
        face_detection_button.grid(row=0, column=3, padx=5, pady=5)

        blur_faces_button = tk.Button(btn_frame, text="Blur Faces", command=self.blur_faces, width=15)
        blur_faces_button.grid(row=1, column=0, padx=5, pady=5)

        emoji_overlay_button = tk.Button(btn_frame, text="Emoji Overlay", command=self.overlay_emojis, width=15)
        emoji_overlay_button.grid(row=1, column=1, padx=5, pady=5)

        emotion_detection_button = tk.Button(btn_frame, text="Emotion Detection", command=self.emotion_detector, width=15)
        emotion_detection_button.grid(row=1, column=2, padx=5, pady=5)
        
        save_image_button = tk.Button(btn_frame, text="Save Image", command=self.save_image, width=15)
        save_image_button.grid(row=1, column=3, padx=5, pady=5)

        # Placeholder for the loaded image and OpenCV image for processing
        self.loaded_image = None
        self.cv_image = None
        self.current_image = None  # Store the currently displayed image

  

    def load_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        # Load and display image
        self.cv_image = cv2.imread(file_path)  # OpenCV image
        self.current_image = self.cv_image.copy()  # Store a copy of the original image
        self.display_image(self.cv_image)

    def display_image(self, cv_img):
        # Resize image for fixed display area (500x500)
        max_size = 500
        h, w = cv_img.shape[:2]
        scale = min(max_size / w, max_size / h)
        resized_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)))

        # Convert OpenCV image (BGR) to Tkinter-compatible image (RGB)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tk_img = ImageTk.PhotoImage(pil_img)

        # Display image on label in fixed area
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img  # Keep reference to avoid garbage collection
        
        # Update current_image with the processed version
        self.current_image = cv_img.copy()


    def open_camera(self):
        # Initialize video capture from the default camera
        cap = cv2.VideoCapture(0)
        
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        image_saved = False  # Flag to track if an image has been saved

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
              # Load and display image
            self.cv_image = frame  # OpenCV image
            self.current_image = self.cv_image.copy()  # Store a copy of the original image
       
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save only the first detected face
                if not image_saved:
                    # Full-frame image with detected face
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                    full_image_path = os.path.join(desktop_path, f"detected_face_{timestamp}.jpg")
                    cv2.imwrite(full_image_path, frame)
                    print(f"Full-frame image saved at: {full_image_path}")
                    
                    # Cropped face-only image
                    face_only = frame[y:y+h, x:x+w]  # Crop the face region from the frame
                    cropped_face_path = os.path.join(desktop_path, f"cropped_face_{timestamp}.jpg")
                    cv2.imwrite(cropped_face_path, face_only)
                    print(f"Cropped face image saved at: {cropped_face_path}")
                    
                    # Set the flag to True after saving both images
                    image_saved = True

            # Display the frame with detected face rectangles
            cv2.imshow("Camera - Press 'q' to quit", frame)
            
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.display_image(self.cv_image)
                break
        
        # Release the capture and close any open windows
        cap.release()
        cv2.destroyAllWindows()



    def convert_to_grayscale(self):
        if self.cv_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        
        gray_img = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        gray_img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        self.cv_image= gray_img_bgr
        self.display_image(self.cv_image)

    def detect_faces(self):
        if self.cv_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        gray_img = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        img_with_faces = self.cv_image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self.display_image(img_with_faces)

    def blur_faces(self):
        if self.cv_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        img_with_blurred_faces = self.cv_image.copy()
        gray_img = cv2.cvtColor(img_with_blurred_faces, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            roi = img_with_blurred_faces[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (55, 55), 0)
            img_with_blurred_faces[y:y+h, x:x+w] = blurred_roi
        
        self.display_image(img_with_blurred_faces)

    def overlay_emojis(self):
        if self.cv_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

             # Open file dialog to select an image
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            if not file_path:
                return
            
            emoji = cv2.imread(file_path)
            if emoji is None:
                messagebox.showerror("Error", "Could not load emoji.png")
                return
            
            img_with_emojis = self.cv_image.copy()
            gray_img = cv2.cvtColor(img_with_emojis, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                emoji_resized = cv2.resize(emoji, (w, h))
                roi = img_with_emojis[y:y+h, x:x+w]
                img_with_emojis[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.5, emoji_resized, 0.5, 0)
            
            self.display_image(img_with_emojis)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in overlay_emojis: {str(e)}")

    def emotion_detector(self):
        if self.cv_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        
        
        # Load pre-trained emotion detection model
        try:
            # Ensure you have a pre-trained model file
            model_path = 'emotion_model.h5'  # You'll need to provide this
            emotion_model = load_model(model_path)
            
            # Emotion labels
            emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            
            # Load Haar Cascade for face detection (using existing method)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Create a copy of the image to draw on
            img_with_emotions = self.cv_image.copy()
            
            for (x, y, w, h) in faces:
                # Extract the face ROI
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to match model's expected input
                roi_gray = roi_gray.astype('float') / 255.0  # Normalize
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                
                # Predict emotion
                preds = emotion_model.predict(roi_gray)
                emotion_index = np.argmax(preds)
                emotion = emotions[emotion_index]
                confidence = preds[0][emotion_index]
                
                # Draw rectangle and emotion text
                cv2.rectangle(img_with_emotions, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(img_with_emotions, label, 
                            (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)
            
            # Display the image with emotion annotations
            self.display_image(img_with_emotions)
            
            # If no faces detected
            if len(faces) == 0:
                messagebox.showinfo("Emotion Detection", "No faces detected in the image.")
        
        except FileNotFoundError:
            messagebox.showerror("Error", "Emotion detection model not found. Please ensure 'emotion_model.h5' is in the correct directory.")
        except Exception as e:
            messagebox.showerror("Emotion Detection Error", str(e))
   
    def save_image(self):
        if self.cv_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        
        # Open file dialog to select save location
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            return
        
        # Save the current image
        cv2.imwrite(file_path, self.cv_image)
        messagebox.showinfo("Image Saved", f"Image saved to: {file_path}")
# Main execution to run the enhanced GUI
root = tk.Tk()
app = EnhancedFaceRecognitionApp(root)
root.mainloop()