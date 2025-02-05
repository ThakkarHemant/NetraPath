import cv2
import numpy as np
import pyttsx3

# Load YOLO model
weights_path = "data/yolov3-tiny.weights"
config_path = "data/yolov3-tiny.cfg"
coco_names_path = "data/coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speech speed

# Load Video (Change to 0 for Webcam)
video_path = "data/video1.mkv"
cap = cv2.VideoCapture(video_path)

# Track previously detected objects to avoid repeating speech
previously_detected = set()

# Define colors for annotations
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends

    height, width, channels = frame.shape

    # YOLO Object Detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process detections
    detected_objects = set()
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                label = classes[class_id]
                detected_objects.add(label)

                # Bounding Box
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                # Draw Rectangle & Label
                color = [int(c) for c in colors[class_id]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Speak newly detected objects only
    new_objects = detected_objects - previously_detected
    if new_objects:
        spoken_text = "I see " + ", ".join(new_objects)
        engine.say(spoken_text)
        engine.runAndWait()
    
    # Update the previously detected objects
    previously_detected = detected_objects

    # Display Output
    cv2.imshow("Navigation Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()