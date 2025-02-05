import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time
from skimage.feature import hog
from skimage import exposure
import pyttsx3
from pathlib import Path

sys.path.append("..")
from learningModels import svm
sys.path.remove("..")

# Constants
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960

MAX_ASPECT = 2
MIN_ASPECT = 0.8
MAX_FILL = 0.99
MIN_FILL = 0.4
MAX_AREA = 1000
MIN_AREA = 50

def check_size(w, h):
    """Check if the dimensions are within acceptable ranges"""
    return MIN_AREA < w * h < MAX_AREA

def check_fill(img, x, y, w, h):
    """Check if the region has appropriate fill ratio"""
    count = 0
    for i in range(h):
        for j in range(w):
            if img[i + y][j + x] == 255:
                count += 1
    return MIN_FILL < count / (w * h) < MAX_FILL

def check_aspect(w, h):
    """Check if the aspect ratio is within acceptable range"""
    return MIN_ASPECT < h / w < MAX_ASPECT

def min_zero(num):
    """Ensure number is not negative"""
    return max(0, num)

def max_num(num, max_num):
    """Ensure number does not exceed maximum"""
    return min(num, max_num)

def color_extraction(img):
    """Extract white and red colors from image"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # White color range
    white_lower_color = np.array([70, 0, 118])
    white_upper_color = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, white_lower_color, white_upper_color)

    # Red color range
    red_lower_color = np.array([0, 161, 32])
    red_upper_color = np.array([196, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower_color, red_upper_color)

    combined_mask = white_mask + red_mask
    combined_img = cv2.bitwise_and(img, img, mask=combined_mask)

    return combined_img, combined_mask

def candidate_extraction(img):
    """Extract potential sign candidates from image"""
    combined_img, combined_mask = color_extraction(img)

    contours_img = combined_mask.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for i in contours:
        bx, by, bw, bh = cv2.boundingRect(i)

        nl = bh + 10
        nx = max_num(min_zero(int(bx + (bw / 2) - (bh / 2)) - 5), 959-nl)
        ny = max_num(min_zero(by - 5), 719-nl)

        if check_size(bw, bh) and check_fill(combined_mask, bx, by, bw, bh) and check_aspect(bw, bh):
            candidates.append([nx, ny, nl])

    for bx, by, bl in candidates:
        cv2.rectangle(img, (bx, by), (bx + bl, by + bl), (0, 255, 0), 2)

    return candidates

def candidate_selection(img, candidates, svm_model):
    """Select and classify candidates using SVM model"""
    features = []
    for x, y, l in candidates:
        resized = cv2.resize(img[y:y + l, x:x + l], (20, 20), interpolation=cv2.INTER_AREA)
        
        b = resized.copy()
        b[:, :, 1:] = 0  # Keep only blue channel

        r = resized.copy()
        r[:, :, 0:2] = 0  # Keep only red channel

        r_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

        red_histogram_feature, _ = hog(r_gray, orientations=8, pixels_per_cell=(5, 5),
                                     cells_per_block=(2, 2), block_norm='L2', visualize=True)
        blue_histogram_feature, _ = hog(b_gray, orientations=8, pixels_per_cell=(5, 5),
                                      cells_per_block=(2, 2), block_norm='L1', visualize=True)

        features.append(np.concatenate((red_histogram_feature, blue_histogram_feature)))

    predictions = []
    for feature in features:
        prediction = svm_model.predict([feature])[0]
        predictions.append(prediction)

    return predictions

class VideoProcessor:
    def __init__(self, video_path):
        """Initialize video processor with video source"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
            
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize SVM model
        self.model = svm.svm()
        self.model.load(0)
        
        # Processing variables
        self.last_frame = None
        self.frame_counter = 0
        self.last_audio_time = time.time()
        self.audio_cooldown = 2.0
        
        # Traffic light detection parameters
        self.light_colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255])
        }
        
    def speak(self, text):
        """Speak text with cooldown"""
        current_time = time.time()
        if current_time - self.last_audio_time >= self.audio_cooldown:
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_audio_time = current_time

    def detect_traffic_light(self, frame):
        """Detect traffic lights and their colors"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_colors = []
        
        for color, (lower, upper) in self.light_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h/w
                    
                    if 1.5 < aspect_ratio < 3.0:
                        detected_colors.append(color)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        return detected_colors

    def get_frame(self):
        """Get video frame with error handling"""
        ret, frame = self.cap.read()
        if not ret:
            return self.last_frame
        self.last_frame = frame
        self.frame_counter += 1
        return frame

    def process_frame(self, frame):
        """Process frame for both traffic lights and signs"""
        if frame is None:
            return []
            
        # Detect traffic lights
        detected_colors = self.detect_traffic_light(frame)
        if detected_colors:
            self.speak(f"Traffic light ahead showing {detected_colors[0]}")
            
        # Process signs
        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        potential_candidates = candidate_extraction(filtered)
        predictions = candidate_selection(filtered, potential_candidates, self.model)
        return predictions

    def interpret_predictions(self, predictions):
        """Interpret sign predictions and provide audio feedback"""
        if not predictions:
            return None
            
        prediction_counts = {1: 0, 2: 0}
        for pred in predictions:
            if pred in prediction_counts:
                prediction_counts[pred] += 1

        if prediction_counts[2] > 0:
            self.speak("Stop sign ahead")
            return "stop"
        elif prediction_counts[1] > 0:
            self.speak("CrossWalk sign ahead")
            return "straight"
        return None

    def run(self):
        """Main processing loop"""
        try:
            cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
            while True:
                frame = self.get_frame()
                if frame is None:
                    print("End of video or error reading frame")
                    break

                predictions = self.process_frame(frame)
                direction = self.interpret_predictions(predictions)
                
                # Only show the main video frame with detections
                cv2.imshow("Traffic Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error during video processing: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.engine.stop()

def main():
    """Main entry point"""
    try:
        processor = VideoProcessor('data/video6.mkv')
        processor.run()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()