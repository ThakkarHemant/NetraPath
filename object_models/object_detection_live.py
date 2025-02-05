import cv2
import torch
import pyttsx3
import time

def object_detection_live(speech_interval=6):
    """Perform live object detection and provide audio descriptions."""
    # Initialize YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

    # Initialize text-to-speech engine
    tts_engine = pyttsx3.init()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the webcam.")

    last_speech_time = time.time() - speech_interval

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]['name'].unique()

        # Generate and speak description
        if detected_objects.size > 0 and (time.time() - last_speech_time) >= speech_interval:
            last_speech_time = time.time()
            description = f"{', '.join(detected_objects)} are visible in the scene."
            print(description)
            tts_engine.say(description)
            tts_engine.runAndWait()

        # Display results
        cv2.imshow('Live Object Detection - Press "c" to Stop', results.render()[0])

        # Exit on 'c' key press
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()