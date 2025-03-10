import time
import lgpio
from picamera2 import Picamera2
import cv2
import os
# PIR sensor setup
PIR_PIN = 21  # Replace with your actual GPIO pin
chip = 0      # Default GPIO chip (change if necessary)

# Initialize the PIR sensor
h = lgpio.gpiochip_open(chip)
lgpio.gpio_claim_input(h, PIR_PIN)

# Specify the folder to save images
output_folder = "/home/tweetypi/feathered-visitors"

# Initialize the camera
camera = Picamera2(tuning="/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json")
camera.preview_configuration.main.size = (1920, 1080)  # Adjust resolution as needed
camera.preview_configuration.main.format = "RGB888"  # RGB format
camera.configure("preview")
camera.start()

print("Waiting for motion... (Press Ctrl+C to exit)")

try:
    while True:
        motion = lgpio.gpio_read(h, PIR_PIN)  # Read the PIR sensor state
        if motion:
            print("Motion Detected!")

            # Capture an image
            frame = camera.capture_array()

            # Save the frame as an image file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"motion_photo_{timestamp}.jpg"
            # Full file path
            file_path = os.path.join(output_folder, file_name)
            cv2.imwrite(file_path, frame)
            print(f"Photo saved as {file_name}")

            # Add a delay to avoid multiple triggers
            time.sleep(2)
        else:
            print("No motion detected.")
            time.sleep(0.5)  # Delay to prevent rapid polling
except KeyboardInterrupt:
    print("\nExiting...")
    lgpio.gpiochip_close(h)
    camera.stop()
