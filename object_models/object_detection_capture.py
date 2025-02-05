import os
from datetime import datetime
import cv2
import torch
from gtts import gTTS
from flask import Flask

app = Flask(__name__, static_folder='static')
app.config['AUDIO_FOLDER'] = 'static/audio'
app.config['CAPTURED_FOLDER'] = 'static/captured'

# Ensure directories exist
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
os.makedirs(app.config['CAPTURED_FOLDER'], exist_ok=True)

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """Adjust brightness and contrast of an image."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def perform_object_detection():
    """Capture an image, perform object detection, and generate an audio description."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the webcam.")

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        cv2.imshow('Press "c" to Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Save captured image
            img_path = os.path.join(app.config['CAPTURED_FOLDER'], 'captured_image.jpg')
            cv2.imwrite(img_path, frame)
            print("Image captured and saved.")

            # Adjust brightness and contrast
            adjusted_frame = adjust_brightness_contrast(frame, alpha=1.5, beta=20)
            adjusted_img_path = os.path.join(app.config['CAPTURED_FOLDER'], 'adjusted_captured_image.jpg')
            cv2.imwrite(adjusted_img_path, adjusted_frame)
            print("Adjusted image saved.")

            # Perform object detection
            results = model(adjusted_frame)
            detected_objects = results.pandas().xyxy[0]['name'].unique()
            description_text = (
                f"The objects detected in the image are: {', '.join(detected_objects)}."
                if detected_objects.size > 0 else "No objects detected."
            )
            print(description_text)

            # Generate audio description
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], f'description_{timestamp}.mp3')
            tts = gTTS(description_text, lang='en')
            tts.save(audio_path)
            print(f"Audio description saved at: {audio_path}")

            cap.release()
            cv2.destroyAllWindows()
            return description_text, os.path.basename(adjusted_img_path), os.path.basename(audio_path)

        elif key == 27:  # Press 'Esc' to exit
            cap.release()
            cv2.destroyAllWindows()
            break

    return None, None, None