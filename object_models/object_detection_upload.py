from gtts import gTTS
import torch
from PIL import Image
import os
from datetime import datetime
from flask import Flask

app = Flask(__name__)
app.config['AUDIO_FOLDER'] = 'static/audio'

# Ensure audio directory exists
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

def detect_objects(image_path):
    """Detect objects in an image and generate an audio description."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}.")

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Load image
    img = Image.open(image_path)

    # Perform object detection
    results = model(img)
    detected_objects = results.pandas().xyxy[0]['name'].unique()

    # Generate description text
    description_text = (
        f"The objects detected in the image are: {', '.join(detected_objects)}."
        if detected_objects.size > 0 else "No objects detected."
    )

    # Generate audio description
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], f'description_{timestamp}.mp3')
    tts = gTTS(description_text, lang='en')
    tts.save(audio_path)

    return description_text, audio_path, timestamp