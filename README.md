# VisionX

A user-friendly Python application for performing various image processing tasks, including face detection, image manipulation, and camera-based operations, all through a graphical user interface (GUI) built with `Tkinter`.

## Features

- **Load Images**: Open and display image files for processing.
- **Open Camera**: Capture live video from your webcam, detect faces, and save images directly to your desktop.
- **Convert to Grayscale**: Transform loaded images into grayscale.
- **Face Detection**: Automatically detect faces in images using Haar cascades.
- **Blur Faces**: Blur detected faces in images to protect privacy.
- **Overlay Emojis**: Add emojis over detected faces in images.
- **Emotion Detection**: Analyze faces to detect emotions like happiness, sadness, anger, and more.
- **Save Images**: Save processed images to your preferred location.

## Requirements

- Python 3.7 or higher
- The following Python libraries:
  - `tkinter`
  - `opencv-python` (`cv2`)
  - `Pillow`
  - `numpy`

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/VisionX.git
   cd VisionX
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python VisionX.py
   ```

## Usage

1. **Launch the application**:
   ```bash
   python VisionX.py
   ```

2. Use the buttons in the GUI to:
   - Load images or open the camera feed.
   - Process images using features like face detection, grayscale conversion, emoji overlay, or emotion detection.
   - Save processed images to your system.

## Folder Structure

```plaintext
VisionX/
├── VisionX.py                    # Main application script
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
```

## Contribution

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your branch.
4. Open a pull request describing your changes.
