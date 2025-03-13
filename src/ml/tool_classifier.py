import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from libs.classify.classify import create_model
from config import Tool, MODEL_PATH, CLASS_NAMES, TARGET_SIZE

class ToolClassifier:
    def __init__(self):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create mapping from class names to Tool enum
        self.class_to_tool = {
            "curved-mayo-scissor": Tool.CURVED_MAYO_SCISSOR,
            "straight-mayo-scissor": Tool.STRAIGHT_MAYO_SCISSOR,
            "scalpel": Tool.SCALPEL,
            "dissection-clamp": Tool.DISSECTION_CLAMP
        }
        
        self.model = None
        
    def initialize(self):
        # Load model
        try:
            self.model = self._load_model(MODEL_PATH)
            print("Tool classifier model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _load_model(self, model_path, num_classes=4):
        model = create_model(num_classes=num_classes, model_name="resnet50").to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()  # Set the model to evaluation mode
        return model
        
    def _preprocess_image(self, image_path):
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL 
        image = Image.fromarray(image)
        
        # Define the transformation pipeline
        transform = transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformations and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def classify_tool_from_photo(self, photo_path):
        if not self.model:
            if not self.initialize():
                raise RuntimeError("Tool classifier not initialized")
                
        try:
            # Check if the file exists
            if not os.path.exists(photo_path):
                raise FileNotFoundError(f"Image file not found: {photo_path}")
                
            # Preprocess the image
            image_tensor = self._preprocess_image(photo_path)
            
            # Perform inference
            with torch.no_grad():
                output = self.model(image_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class_idx = predicted.item()
            
            # Get the class name
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            print(f"Classified tool: {predicted_class_name}")
            
            # Convert to Tool enum
            tool = self.class_to_tool[predicted_class_name]
            return tool
            
        except Exception as e:
            print(f"Error during classification: {e}")
            raise  # Re-raise the exception to be handled by the caller