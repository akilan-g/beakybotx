import cv2
import lgpio
import time

# GPIO setup for PIR sensor
PIR_PIN = 21  # GPIO pin connected to PIR sensor
h = lgpio.gpiochip_open(0)  # Open GPIO chip 0
lgpio.gpio_claim_input(h, PIR_PIN)  # Set the PIR_PIN as an input pin

# Initialize OpenCV camera
camera = cv2.VideoCapture(0)  # Adjust the index if using a different camera

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Waiting for motion... (Press Ctrl+C to exit)")

try:
    while True:
        motion = lgpio.gpio_read(h, PIR_PIN)  # Read the PIR sensor state
        if motion:
            print("Motion Detected!")
            
            # Capture a frame
            ret, frame = camera.read()
            if ret:
                # Save the frame as an image file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_name = f"motion_photo_{timestamp}.jpg"
                cv2.imwrite(file_name, frame)
                print(f"Photo saved as {file_name}")
            else:
                print("Error: Could not capture the frame.")
            
            # Add a delay to avoid multiple triggers
            time.sleep(2)
        else:
            print("No motion detected.")
        
        time.sleep(0.5)  # Check the PIR sensor state every 0.5 seconds

except KeyboardInterrupt:
    print("\nExiting...")
    lgpio.gpiochip_close(h)  # Clean up GPIO resources
    camera.release()  # Release the camera
