import cv2
from gtts import gTTS
import os
from datetime import datetime
from flask import Flask, render_template, request, Response
import easyocr  # Replacement for pytesseract

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/captured'
app.config['AUDIO_FOLDER'] = 'static/audio'

def perform_ocr_and_audio(image_path):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # For English text
    
    # Load and process image
    img = cv2.imread(image_path)
    
    def adjust_brightness_contrast(image, alpha=1.0, beta=0):
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    # Enhance image quality
    img_adjusted = adjust_brightness_contrast(img, alpha=1.5, beta=20)
    
    # Save adjusted image
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    adjusted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'adjusted_image_{timestamp}.jpg')
    cv2.imwrite(adjusted_image_path, img_adjusted)

    # Perform OCR using EasyOCR
    results = reader.readtext(img_adjusted)
    
    # Extract text from results
    text = " ".join([result[1] for result in results])  # result[1] contains the text
    
    print("Recognized text:", text)

    # Generate audio
    tts = gTTS(text=text, lang='en')
    audio_file = os.path.join(app.config['AUDIO_FOLDER'], f'output_{timestamp}.mp3')
    tts.save(audio_file)

    return text, audio_file, adjusted_image_path, timestamp