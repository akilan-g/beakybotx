import io  # Import io for input/output operations (not used directly in this version)
import cv2  # Import OpenCV for image processing and encoding
from flask import Flask, Response  # Import Flask for web server and Response for streaming
from picamera2 import Picamera2  # Import Picamera2 for controlling the Raspberry Pi camera
import time
import numpy as np
import tensorflow as tf

app = Flask(__name__) # Create a Flask web application

# Load the TFLite model for classification
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get model input shape
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]

# Define class names
class_names = ['Black_Footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross']

def preprocess_image(image):
    """Preprocess the image to match model requirements"""
    # Resize the image to match the model input shape
    resized = cv2.resize(image, (input_width, input_height))
    
    # Convert BGR to RGB (TensorFlow models typically expect RGB)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize the image
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    batch_image = np.expand_dims(normalized, axis=0)
    
    return batch_image

def run_inference(image):
    """Run inference on the image using TFLite model"""
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

def detect_bird_location(frame):
    """Detect where the bird is located in the image using image processing"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to find potential bird
    min_contour_area = 500  # Minimum area to be considered (adjust based on your images)
    potential_birds = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    # Sort by area (largest first)
    potential_birds.sort(key=cv2.contourArea, reverse=True)
    
    # If no suitable contours, return full image bounding box
    if not potential_birds:
        height, width = frame.shape[:2]
        return (0, 0, width, height)
    
    # Take the largest contour as the bird
    x, y, w, h = cv2.boundingRect(potential_birds[0])
    
    # Add some padding around the bounding box
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2*padding)
    h = min(frame.shape[0] - y, h + 2*padding)
    
    return (x, y, w, h)

def generate_frames(): # Function to generate frames continuously for video streaming
#with Picamera2() as camera: # Initialize the Raspberry Pi camera using Picamera2
    camera=Picamera2(tuning="/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json")       
    camera.preview_configuration.main.size = (1920, 1080)  # Set the camera resolution to mentioned pixels
    camera.preview_configuration.main.format = "RGB888" # Set the camera image format to RGB888 (24-bit color)
    camera.preview_configuration.align()  # Align the camera's preview configuration (important for hardware setup)
    camera.configure("preview")  # Configure the camera to be in preview mode
    
    camera.start()       # Start the camera to begin capturing video
    while True:
        
        frame = camera.capture_array()  # Capture a frame from the camera as a NumPy array
        
        # Process the frame for bird detection
        start_time = time.time()
        
        # Preprocess the frame
        processed_image = preprocess_image(frame)
        
        # Run bird classification
        predictions = run_inference(processed_image)
        
        # Get the predicted class and confidence
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        predicted_class = class_names[class_idx]
        
        # Find bird location
        x, y, w, h = detect_bird_location(frame)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text overlay for class and confidence
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{predicted_class}: {confidence:.2f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add inference time
        inference_time = (time.time() - start_time) * 1000
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])      # Encode the captured frame as a JPEG image
        if not ret:  # If encoding fails, skip to the next iteration
            continue
        frame_bytes = buffer.tobytes()    # Convert the encoded frame into bytes format for transmission
        # Yield the frame in the correct format for live streaming
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # Add a delay to control frame rate (30 frames per second)

@app.route('/video_feed') # Define the route for video streaming (/video_feed) in the Flask app
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Return the video stream response using the 'generate_frames' function

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Bird Detection Stream</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
          h1 { color: #2c3e50; }
          img { max-width: 100%; border: 2px solid #3498db; border-radius: 5px; }
        </style>
      </head>
      <body>
        <h1>Bird Species Detection</h1>
        <img src="/video_feed" width="1280" height="720" />
      </body>
    </html>
    """

if __name__ == '__main__': # Run the Flask web server if the script is being executed directly
# The Flask server will run on all available network interfaces ('0.0.0.0')
# The server will run on port 5000 and allow multiple threads to handle requests simultaneously
    app.run(host='0.0.0.0', port=5000, threaded=True)