import time
import lgpio
from picamera2 import Picamera2
import cv2

# PIR sensor setup
PIR_PIN = 21  # Replace with your actual GPIO pin
chip = 0      # Default GPIO chip (change if necessary)

# Initialize the PIR sensor
h = lgpio.gpiochip_open(chip)
lgpio.gpio_claim_input(h, PIR_PIN)

# Initialize the camera
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)  # Adjust resolution as needed
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
            cv2.imwrite(file_name, frame)
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
