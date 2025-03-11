import os
import sys
import string
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'libs', 'WhisperLive'))

import google.generativeai as genai
import serial
from whisper_live.client import TranscriptionClient
from dotenv import load_dotenv
from enum import Enum

class EnvironmentLoader:
    @staticmethod
    def load():
        load_dotenv()  
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            print("Error: GOOGLE_API_KEY environment variable not set. Exiting.")
            exit()
        return gemini_key

class CommandType(Enum):
    DISPENSE = "dispense"
    RETURN = "return"
    NOT_A_REQUEST = "not_a_request"

class Tool(Enum):
    STRAIGHT_MAYO_SCISSOR = "straight mayo scissor"
    CURVED_MAYO_SCISSOR = "curved mayo scissor"
    SCALPEL = "scalpel"
    DISSECTION_CLAMP = "dissection clamp"

class ToolInventory:
    def __init__(self):
        self.inventory = {tool: True for tool in Tool}  # True means the tool is in, False means the tool is out

    def is_tool_in(self, tool):
        return self.inventory.get(tool, False)

    def dispense_tool(self, tool):
        if self.is_tool_in(tool):
            self.inventory[tool] = False
            return True
        return False

    def return_tool(self, tool):
        if not self.is_tool_in(tool):
            self.inventory[tool] = True
            return True
        return False
    
class Command:
    def __init__(self, command_string):
        if command_string == "not a request":
            self.tool = None
            self.command_type = CommandType.NOT_A_REQUEST
            return
        
        parts = command_string.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid command string: {command_string}")
        
        tool, command = parts
        self.tool = Tool(tool)
        self.command_type = CommandType(command)

    def __str__(self):
        return f"{self.tool.value} {self.command_type.value}"

class ToolLocations:
    DISPENSE_LOCATIONS = {
        Tool.STRAIGHT_MAYO_SCISSOR: "D1",
        Tool.CURVED_MAYO_SCISSOR: "D2",
        Tool.SCALPEL: "D3",
        Tool.DISSECTION_CLAMP: "D4"
    }

    RETURN_LOCATIONS = {
        Tool.STRAIGHT_MAYO_SCISSOR: "R1",
        Tool.CURVED_MAYO_SCISSOR: "R2",
        Tool.SCALPEL: "R3",
        Tool.DISSECTION_CLAMP: "R4"
    }

    @staticmethod
    def get_location(command: Command):
        if command.command_type == CommandType.DISPENSE:
            return ToolLocations.DISPENSE_LOCATIONS[command.tool]
        elif command.command_type == CommandType.RETURN:
            return ToolLocations.RETURN_LOCATIONS[command.tool]
        else:
            raise ValueError(f"Invalid command type: {command.command_type}")
        
class CameraManager:
    PHOTO_PATH = '/home/orobot/orobot/src/photo.jpg'
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            exit()

    def take_photo(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(self.PHOTO_PATH, frame)
            print("Photo taken and saved as photo.jpg")
        else:
            print("Error: Could not read frame from camera.")
        self.cap.release()
        
class ToolClassifier:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'classify', 'tool_classifier.pth')
    CLASS_NAMES = ["curved-mayo-scissor", "dissection-clamp", "scalpel", "straight-mayo-scissor"]
    TARGET_SIZE = (256, 144)
    
    def __init__(self):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = self.load_model(self.MODEL_PATH)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Create mapping from class names to Tool enum
        self.class_to_tool = {
            "curved-mayo-scissor": Tool.CURVED_MAYO_SCISSOR,
            "straight-mayo-scissor": Tool.STRAIGHT_MAYO_SCISSOR,
            "scalpel": Tool.SCALPEL,
            "dissection-clamp": Tool.DISSECTION_CLAMP
        }
    
    def load_model(self, model_path, num_classes=4):
        model = create_model(num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()  # Set the model to evaluation mode
        return model
        
    def preprocess_image(self, image_path):
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Define the transformation pipeline
        transform = transforms.Compose([
            transforms.Resize(self.TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformations and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def classify_tool_from_photo(self, photo_path):
        try:
            # Check if the file exists
            if not os.path.exists(photo_path):
                raise FileNotFoundError(f"Image file not found: {photo_path}")
                
            # Preprocess the image
            image_tensor = self.preprocess_image(photo_path)
            
            # Perform inference
            with torch.no_grad():
                output = self.model(image_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class_idx = predicted.item()
            
            # Get the class name
            predicted_class_name = self.CLASS_NAMES[predicted_class_idx]
            print(f"Classified tool: {predicted_class_name}")
            
            # Convert to Tool enum
            tool = self.class_to_tool[predicted_class_name]
            return tool
            
        except Exception as e:
            print(f"Error during classification: {e}")
            raise  # Re-raise the exception to be handled by the caller
    
class CommandClassifier:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')

    def classify(self, transcribed_text) -> Command:
        transcribed_text = transcribed_text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

        prompt = f"""
        Prompt:
        You are an AI that extracts valid tool requests from transcribed speech. Your task is to determine whether the transcription contains a command to dispense or return a tool.

        ### Command Requirements:
        1. The phrase **must contain** the word **"robot"** (case-insensitive).
        2. The word **"dispense"** or **"return"** must follow "robot" at some point in the transcription.
        3. One of the following tool names must appear after the command:  
        {Tool.TOOLS}

        Ignore extra words before "robot" or between the required words. Commands may have minor typos or extra punctuation. If a valid command is repeated multiple times, process it as a single request.

        ### Response Format:
        - If a valid command is detected, return:  
        **`"<tool> <command>"`** (all lowercase, separated by a space).  
        - If no valid command is found, return exactly:  
        **"not a request"**  

        ### Examples:

        **Input:** "robot, please dispense the scalpel."  
        **Output:** "scalpel dispense"  

        **Input:** "robot return curved mayo scissor!"  
        **Output:** "curved mayo scissor return"  

        **Input:** "Yeah robot dispense scalpel"  
        **Output:** "scalpel dispense"  

        **Input:** "Yeah, robot dispense scalpel robot dispense scalpel"  
        **Output:** "scalpel dispense"  

        **Input:** "robot retrieve the dissection clamp"  
        **Output:** "not a request"  

        **Input:** "dispense scalpel, robot."  
        **Output:** "not a request"  

        Process the following transcription accordingly:  
        "{transcribed_text}"
        """
        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip().lower()
            return Command(result)
        except Exception as e:
            print(f"Error during API call: {e}")
            return Command("not a request")
        

class SerialCommunicator:
    DEVICE_NAME = "/dev/ttyUSB0"

    def __init__(self):
        self.ser = serial.Serial(self.DEVICE_NAME, 9600)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send(self, location: str):
        self.ser.write(location.encode('utf-8'))
        # debug
        while self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').rstrip()
            print(line)

class ControlService:
    def __init__(self):
        self.GEMINI_KEY = EnvironmentLoader.load()
        self.tool_classifier = CommandClassifier(self.GEMINI_KEY)
        self.camera_manager = CameraManager()
        self.serial_communicator = SerialCommunicator()
        self.tool_photo_classifier = ToolClassifier()
        self.tool_inventory = ToolInventory()

        # Initialize Transcription Client
        self.client = TranscriptionClient(
            host="localhost",
            port=9090,
            lang="en",
            translate=False,
            model="tiny.en",
            use_vad=True,
            callback=self.handle_transcription
        )

        self.last_text = ""

    def handle_transcription(self, text, is_final):
        if is_final and text != self.last_text:
            self.last_text = text
            self.client.client.paused = True

            # Send the transcribed text to the LLM
            response = self.tool_classifier.classify(text[-1])
            print(f"Transcribed Text: {text[-1]}\n LLM Response: {response}")
            self.execute_command(response)
            self.client.client.paused = False

    def execute_command(self, command: Command):
        if command.command_type != CommandType.NOT_A_REQUEST:
            if command.command_type == CommandType.DISPENSE:
                if self.tool_inventory.dispense_tool(command.tool):
                    location = ToolLocations.get_location(command)
                    self.serial_communicator.send(location)
                else:
                    print(f"Error: Tool {command.tool.value} is already dispensed.")
            elif command.command_type == CommandType.RETURN:
                self.camera_manager.take_photo()
                try:
                    # Classify the photo to determine the tool
                    classified_tool = self.tool_photo_classifier.classify_tool_from_photo(CameraManager.PHOTO_PATH)
                    print(f"Classified tool: {classified_tool.value}")
                    
                    # Check if the classified tool is already marked as returned in inventory
                    if self.tool_inventory.is_tool_in(classified_tool):
                        print(f"Warning: Classified tool {classified_tool.value} is already marked as in.")
                        
                        # Find a tool that's actually out to return instead
                        out_tool = None
                        for tool in Tool:
                            if not self.tool_inventory.is_tool_in(tool):
                                out_tool = tool
                                break
                        
                        if out_tool:
                            print(f"Using {out_tool.value} instead since it's marked as out.")
                            command.tool = out_tool
                        else:
                            print("No tools are marked as out. Using classified tool anyway.")
                            command.tool = classified_tool
                    else:
                        # Normal case - the classified tool is out according to inventory
                        command.tool = classified_tool
                    
                    # Now return the selected tool
                    if self.tool_inventory.return_tool(command.tool):
                        location = ToolLocations.get_location(command)
                        self.serial_communicator.send(location)
                        print(f"Tool {command.tool.value} successfully returned.")
                    else:
                        print(f"Error: Tool {command.tool.value} is already returned.")
                        
                except Exception as e:
                    print(f"Tool classification failed: {e}")
                    # Even if classification fails, try to return a tool if one is out
                    out_tool = None
                    for tool in Tool:
                        if not self.tool_inventory.is_tool_in(tool):
                            out_tool = tool
                            break
                    
                    if out_tool:
                        print(f"Classification failed but found tool marked as out: {out_tool.value}")
                        command.tool = out_tool
                        self.tool_inventory.return_tool(command.tool)
                        location = ToolLocations.get_location(command)
                        self.serial_communicator.send(location)
                    else:
                        print("Classification failed and no tools are marked as out. Cannot proceed.")
                        
    def start(self):
        # Start the transcription client
        self.client()

if __name__ == "__main__":
    control_service = ControlService()
    # serial_comm = SerialCommunicator()
    # while True:
    #     if serial_comm.ser.in_waiting > 0:
    #         line = serial_comm.ser.readline().decode('utf-8').rstrip()
    #         print(line)
    #         if line == "Homing complete. Waiting for commands (D1-D4, R1-R4)":
    #             break
    
    control_service.start()