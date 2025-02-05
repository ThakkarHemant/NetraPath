import cv2
import easyocr
from gtts import gTTS
import os
from datetime import datetime

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    # Apply brightness and contrast adjustment
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_img

def perform_text_capture():
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press "c" to Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Create directories if they don't exist
            os.makedirs('static/captured', exist_ok=True)
            os.makedirs('static/audio', exist_ok=True)
            
            # Save the original image
            cv2.imwrite('static/captured/original_image.jpg', frame)
            
            # Adjust brightness and contrast
            adjusted_frame = adjust_brightness_contrast(frame, alpha=1.5, beta=20)
            cv2.imwrite('static/captured/adjusted_image.jpg', adjusted_frame)
            
            cap.release()
            cv2.destroyAllWindows()
            print("Images captured and saved")
            break
            
        elif key == 27:  # Press 'Esc' to exit
            cap.release()
            cv2.destroyAllWindows()
            return "Capture cancelled", "", "", ""
    
    img_path = 'static/captured/adjusted_image.jpg'
    if os.path.exists(img_path):
        # Read the image
        img = cv2.imread(img_path)
        
        # Perform OCR using EasyOCR
        results = reader.readtext(img)
        
        # Extract text from results
        text_output = ' '.join([result[1] for result in results])
        
        print("Recognized text:")
        print(text_output)
        
        if text_output.strip():  # Check if the recognized text is not empty
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Convert recognized text to speech using gTTS
            tts = gTTS(text=text_output, lang='en')
            audio_path = f"static/audio/output_audio_{timestamp}.mp3"
            tts.save(audio_path)
            
            return text_output, img_path, audio_path, timestamp
        else:
            return "Error: No text detected in the captured image", "", "", ""
    else:
        return "Error: Adjusted image not found", "", "", ""