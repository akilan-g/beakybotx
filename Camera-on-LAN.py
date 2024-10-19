import io  # Import io for input/output operations (not used directly in this version)
import cv2  # Import OpenCV for image processing and encoding
from flask import Flask, Response  # Import Flask for web server and Response for streaming
from picamera2 import Picamera2  # Import Picamera2 for controlling the Raspberry Pi camera

# Create a Flask web application
app = Flask(__name__)

# Function to generate frames continuously for video streaming
def generate_frames():
    # Initialize the Raspberry Pi camera using Picamera2
    with Picamera2() as camera:
        # Set the camera resolution to 640x480 pixels
        camera.preview_configuration.main.size = (640, 480)
        
        # Set the camera image format to RGB888 (24-bit color)
        camera.preview_configuration.main.format = "RGB888"
        
        # Align the camera's preview configuration (important for hardware setup)
        camera.preview_configuration.align()
        
        # Configure the camera to be in preview mode
        camera.configure("preview")
        
        # Start the camera to begin capturing video
        camera.start()

        # Infinite loop to continuously capture frames
        while True:
            # Capture a frame from the camera as a NumPy array
            frame = camera.capture_array()

            # Encode the captured frame as a JPEG image
            ret, buffer = cv2.imencode('.jpg', frame)
            
            # If encoding fails, skip to the next iteration
            if not ret:
                continue

            # Convert the encoded frame into bytes format for transmission
            frame_bytes = buffer.tobytes()

            # Yield the frame in the correct format for live streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Define the route for video streaming (/video_feed) in the Flask app
@app.route('/video_feed')
def video_feed():
    # Return the video stream response using the 'generate_frames' function
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask web server if the script is being executed directly
if __name__ == '__main__':
    # The Flask server will run on all available network interfaces ('0.0.0.0')
    # The server will run on port 5000 and allow multiple threads to handle requests simultaneously
    app.run(host='0.0.0.0', port=5000, threaded=True)
