import lgpio
import time

# Set up the GPIO pin where your PIR sensor is connected
PIR_PIN = 21  # Replace with the actual GPIO pin number you're using
chip = 0
# Open the GPIO chip and configure the PIR pin
h = lgpio.gpiochip_open(chip)  # Open GPIO chip 0
lgpio.gpio_claim_input(h, PIR_PIN)  # Set PIR_PIN as input

print("Starting PIR sensor monitoring... (Press Ctrl+C to exit)")

try:
    while True:
        motion_detected = lgpio.gpio_read(h, PIR_PIN)  # Read the PIR sensor pin
        if motion_detected:
            print("Motion Detected!")
        else:
            print("No Motion")
        time.sleep(0.5)  # Add a small delay to avoid rapid outputs
except KeyboardInterrupt:
    print("\nExiting...")
    lgpio.gpiochip_close(chip_handle)  # Clean up GPIO settings on exit

