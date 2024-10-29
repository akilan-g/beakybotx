import io  # Import io for input/output operations (not used directly in this version)
import cv2  # Import OpenCV for image processing and encoding
from flask import Flask, Response  # Import Flask for web server and Response for streaming
from picamera2 import Picamera2  # Import Picamera2 for controlling the Raspberry Pi camera
import time 

app = Flask(__name__) # Create a Flask web application

def generate_frames(): # Function to generate frames continuously for video streaming

#with Picamera2() as camera: # Initialize the Raspberry Pi camera using Picamera2
    camera=Picamera2(tuning="/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json")       
    camera.preview_configuration.main.size = (320, 240)  # Set the camera resolution to 640x480 pixels
    camera.preview_configuration.main.format = "RGB888" # Set the camera image format to RGB888 (24-bit color)
    camera.preview_configuration.align()  # Align the camera's preview configuration (important for hardware setup)
    camera.configure("preview")  # Configure the camera to be in preview mode
    
    camera.start()       # Start the camera to begin capturing video

    while True:
        
        frame = camera.capture_array()  # Capture a frame from the camera as a NumPy array
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])      # Encode the captured frame as a JPEG image
        if not ret:  # If encoding fails, skip to the next iteration
            continue
        frame_bytes = buffer.tobytes()    # Convert the encoded frame into bytes format for transmission
        # Yield the frame in the correct format for live streaming
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)  # Add a delay to control frame rate (10 frames per second)


@app.route('/video_feed') # Define the route for video streaming (/video_feed) in the Flask app
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Return the video stream response using the 'generate_frames' function

if __name__ == '__main__': # Run the Flask web server if the script is being executed directly
# The Flask server will run on all available network interfaces ('0.0.0.0')
# The server will run on port 5000 and allow multiple threads to handle requests simultaneously
    app.run(host='0.0.0.0', port=5000, threaded=True)
