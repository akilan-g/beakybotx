import cv2
import numpy as np
#from tensorflow.lite.python.interpreter import Interpreter as tflite
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
# Load the TFLite model
model_path = "/home/tweetypi/detect.tflite"
label_path = "/home/tweetypi/labelmap.txt"

# Load labels
with open(label_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the TFLite interpreter
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
input_shape = input_details[0]['shape']

# Initialize the camera (you can replace with your camera index or PiCamera setup)
#cap = cv2.VideoCapture(0)  # 0 for the default camera
cap=Picamera2()       
cap.preview_configuration.main.size = (1920, 1080)  # Set the camera resolution to mentioned pixels
cap.preview_configuration.main.format = "RGB888" # Set the camera image format to RGB888 (24-bit color)
cap.preview_configuration.align()  # Align the camera's preview configuration (important for hardware setup)
cap.configure("preview")  # Configure the camera to be in preview mode
print("Starting the object detection... Press 'q' to exit.")

cap.start()
#picam2.start()

print("Starting the object detection... Press 'q' to exit.")

#while cap.isOpened():
    #ret, frame = cap.read()
    #if not ret:
        #print("Failed to grab frame")
        #break


while True:
    # Capture frame
    frame = cap.capture_array()  # Get the image as a NumPy array
    
    if frame is None:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    # Process detection results
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = int(classes[i])
            label = labels[class_id]
            if label == "bird":  # Focus on detecting birds
		        print("bird detected")
                    ymin, xmin, ymax, xmax = boxes[i]
                    h, w, _ = frame.shape
                    xmin = int(xmin * w)
                    xmax = int(xmax * w)
                    ymin = int(ymin * h)
                    ymax = int(ymax * h)

                # Draw bounding box and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({int(scores[i] * 100)}%)", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Bird Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
