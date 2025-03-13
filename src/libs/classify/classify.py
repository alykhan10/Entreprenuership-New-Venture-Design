import os
import sys

import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import cv2  # Import OpenCV
import time

try:
    from model import create_model  # Try direct import first
except ImportError:
    # If that fails, try to add the directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import create_model


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model(model_path, num_classes=4, model_name="resnet50"):
    model = create_model(num_classes=num_classes, model_name=model_name).to(device)
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
    model_path = "tool_classifier_new.pth"
    
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

# Add this function to your classify.py file
def interactive_test():
    """
    Interactive testing mode with visualization for Windows.
    This function captures frames from your camera and displays them with predictions.
    Press 'q' to exit, 's' to save the current frame.
    """
    # Path to the trained model
    model_path = "tool_classifier_new.pth"
    
    # Load the model
    model = load_model(model_path, model_name="resnet50")
    
    # Define class names
    class_names = ["curved-mayo-scissor", "dissection-clamp", "scalpel", "straight-mayo-scissor"]
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    # Set the resolution to 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Interactive test mode:")
    print("- Press 'q' to exit")
    print("- Press 's' to save the current frame with its prediction")
    
    # Create a directory for saving images if it doesn't exist
    save_dir = "captured_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Counter for prediction stability
    last_predictions = []
    max_predictions = 5  # Number of predictions to average
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            # Create a copy of the frame for displaying
            display_frame = frame.copy()
            
            # Classify the captured image
            predicted_class = classify_image(model, frame, class_names)
            
            # Add to prediction history for stability
            last_predictions.append(predicted_class)
            if len(last_predictions) > max_predictions:
                last_predictions.pop(0)
            
            # Get most common prediction from history
            from collections import Counter
            most_common = Counter(last_predictions).most_common(1)[0][0]
            
            # Display the result on the frame
            cv2.putText(display_frame, f"Prediction: {most_common}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Camera Feed with Classification', display_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save the image with timestamp and prediction
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{save_dir}/{timestamp}_{most_common}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved as {filename}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

# Modify your main function to add an option to run interactive test
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tool Classification')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode with visualization')
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test()
    else:
        main()