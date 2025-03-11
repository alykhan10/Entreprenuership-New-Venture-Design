import os
import sys

import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .model import create_model  # Import your model creation function
import cv2  # Import OpenCV
import time



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model(model_path, num_classes=4):
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input image
def preprocess_image(image, target_size=(256, 144)):
    # Convert the OpenCV image (BGR) to PIL image (RGB)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize to match training size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    # Apply transformations and add a batch dimension
    image = transform(image).unsqueeze(0).to(device)
    return image

# Classify the image
def classify_image(model, image, class_names):
    # Preprocess the image
    image_tensor = preprocess_image(image)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
    
    # Map the predicted class index to the class name
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

# Main function to capture and classify images
def main():
    # Path to the trained model
    model_path = "tool_classifier.pth"
    
    # Load the model
    model = load_model(model_path)
    
    # Define class names (replace with your actual class names)
    class_names = ["curved-mayo-scissor", "dissection-clamp", "scalpel", "straight-mayo-scissor"]
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    # Set the resolution to 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Running in headless mode.")
    print("Press Ctrl+C to exit.")
    
    # Create a directory for saving images if it doesn't exist
    save_dir = "captured_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            # Classify the captured image
            predicted_class = classify_image(model, frame, class_names)
            print(f"Predicted Class: {predicted_class}")
            
            # Save the image with timestamp and prediction
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{save_dir}/{timestamp}_{predicted_class}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            
            # Wait before the next capture
            time.sleep(5)  # Capture every 5 seconds
    
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
    
    # Release the camera
    cap.release()

# Run the main function
if __name__ == "__main__":
    main()