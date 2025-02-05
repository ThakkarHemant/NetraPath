import cv2
import easyocr
from gtts import gTTS
import os
from datetime import datetime
import threading
import pygame
import time

class LiveTextRecognition:
    def __init__(self):
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'])
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        # Control flag for the recognition loop
        self.running = True
        # Store last recognized text to avoid repetition
        self.last_text = ""
        # Minimum length for text to be considered valid
        self.min_text_length = 3
        # Cooldown period for text-to-speech (in seconds)
        self.speech_cooldown = 2.0
        self.last_speech_time = 0
        # Lock for thread-safe audio handling
        self.audio_lock = threading.Lock()
        # Buffer for text stability
        self.text_buffer = []
        self.buffer_size = 5
        # Window state
        self.window_closed = False

    def generate_audio(self, text):
        """Generate and play audio for detected text"""
        if not text.strip():
            return

        current_time = time.time()
        if current_time - self.last_speech_time < self.speech_cooldown:
            return

        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            audio_path = os.path.join('static/audio', f'live_text_{timestamp}.mp3')
            
            with self.audio_lock:
                tts = gTTS(text=text, lang='en')
                tts.save(audio_path)
                
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Clean up audio file after playing
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                
            self.last_speech_time = current_time
        except Exception as e:
            print(f"Error in audio generation/playback: {e}")

    def get_stable_text(self, new_text):
        """Use a buffer to stabilize text recognition"""
        if len(new_text) < self.min_text_length:
            return ""

        self.text_buffer.append(new_text)
        if len(self.text_buffer) > self.buffer_size:
            self.text_buffer.pop(0)

        # Only return text if it appears in majority of recent readings
        if len(self.text_buffer) >= 3:
            # Count occurrences of each text version
            text_counts = {}
            for text in self.text_buffer:
                text_counts[text] = text_counts.get(text, 0) + 1
            
            # Find most common text
            most_common = max(text_counts.items(), key=lambda x: x[1])
            if most_common[1] >= 2:  # At least 2 occurrences
                return most_common[0]

        return ""

    def process_frame(self, frame):
        """Process a single frame and return recognized text"""
        try:
            results = self.reader.readtext(frame)
            text = ' '.join([result[1] for result in results])
            
            # Draw boxes around detected text
            for (bbox, text_part, prob) in results:
                if prob > 0.2:  # Only show confident detections
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    top_left = tuple(map(int, top_left))
                    bottom_right = tuple(map(int, bottom_right))
                    
                    # Draw box
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    # Add text above the box
                    cv2.putText(frame, text_part, (top_left[0], top_left[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            stable_text = self.get_stable_text(text)
            return frame, stable_text
        except Exception as e:
            print(f"Error in text recognition: {e}")
            return frame, ""

def perform_live_text():
    """Main function to perform live text recognition"""
    recognizer = LiveTextRecognition()
    cap = cv2.VideoCapture(0)
    
    try:
        while recognizer.running and not recognizer.window_closed:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame, text = recognizer.process_frame(frame)
            
            # If stable text is detected and it's different from last text
            if text and text != recognizer.last_text:
                recognizer.last_text = text
                # Start a new thread for audio generation
                threading.Thread(target=recognizer.generate_audio, args=(text,)).start()
            
            # Display the frame
            cv2.imshow('Live Text Recognition (Press Q to quit)', processed_frame)
            
            # Check for window close or 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('Live Text Recognition (Press Q to quit)', cv2.WND_PROP_VISIBLE) < 1:
                recognizer.window_closed = True
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        return "Live text recognition stopped"
    
    except Exception as e:
        return f"Error: {str(e)}"
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        recognizer.running = False

if __name__ == "__main__":
    perform_live_text()