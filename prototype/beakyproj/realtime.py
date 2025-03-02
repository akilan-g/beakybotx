import cv2
import numpy as np
import tensorflow as tf
import time

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
    
    return batch_image, resized

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

def process_frame(frame):
    """Process a frame through the model and return annotated frame"""
    # Copy the frame for display
    display_frame = frame.copy()
    
    # Preprocess the entire frame for classification
    processed_image, _ = preprocess_image(frame)
    
    # Run classification inference
    predictions = run_inference(processed_image)
    
    # Get the predicted class index and confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    predicted_class = class_names[class_idx]
    
    # Find bird location in the image
    x, y, w, h = detect_bird_location(frame)
    
    # Draw bounding box around the bird
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw rectangle at the top for text background
    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
    
    # Add prediction text
    prediction_text = f"{predicted_class}: {confidence:.2f}%"
    cv2.putText(display_frame, prediction_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display_frame

def main():
    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Adjust camera resolution if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit")
    
    # Process frames in a loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Process the frame
        start_time = time.time()
        result_frame = process_frame(frame)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Add inference time text
        cv2.putText(result_frame, f"Inference: {inference_time:.1f}ms", 
                   (10, result_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('TFLite Bird Detection', result_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()