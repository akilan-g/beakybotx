import cv2  # Importing OpenCV for image display and processing
from picamera2 import Picamera2  # Importing Picamera2 to control the Raspberry Pi Camera

camera = Picamera2() # Initialize the camera
camera.preview_configuration.main.size = (360, 360) # Set the camera resolution to 360x360
camera.preview_configuration.main.format = "RGB888" # Set the camera format to RGB888 (RGB with 8 bits per channel)
camera.preview_configuration.align() # Align the camera configuration (ensures the configuration is properly set)
camera.configure("preview") # Apply the configuration to set the camera into preview mode
camera.start() # Start the camera to begin capturing frames

while True: 
 
    frame = camera.capture_array()    # Capture a frame from the camera as an array

    cv2.imshow("camera", frame) # Display the captured frame in a window named "camera"

    if cv2.waitKey(1) == ord('q'):     # Check if the 'q' key is pressed, if yes, break the loop and stop capturing
        
        break

cv2.destroyAllWindows() # Close all OpenCV windows when the loop is terminated
