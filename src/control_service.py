from config import Tool, CommandType, PHOTO_PATH
from models.tools import ToolInventory
from models.commands import Command, ToolLocations
from services.camera_service import CameraManager
from services.serial_service import SerialCommunicator
from services.voice_service import EnvironmentLoader, CommandClassifier, VoiceTranscriptionService
from ml.tool_classifier import ToolClassifier
import threading
import time
import os
import sys

# Add web directory to path to import web app
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web'))
from app import run_web_server, shared_state, get_command, emit_status_update

class ControlService:
    def __init__(self):
        # Load API key
        self.api_key = EnvironmentLoader.load()
        if not self.api_key:
            raise ValueError("Failed to load API key")
        
        self.command_classifier = CommandClassifier(self.api_key)
        self.camera_manager = CameraManager()
        self.serial_communicator = SerialCommunicator()
        self.tool_classifier = ToolClassifier()
        self.tool_inventory = ToolInventory()
        self.voice_service = VoiceTranscriptionService(self._process_transcription)

        self.last_command_type = None
        
        # Update shared state with initial inventory
        shared_state.update_inventory(self.tool_inventory)
    
    def _process_transcription(self, text):
        """Process transcribed text"""
        print(f"Processing transcription: {text}")
        # Update the shared state with the current transcript
        shared_state.update_transcript(text)
        emit_status_update()
        
        command = self.command_classifier.classify_text(text)
        self.execute_command(command)
    
    def execute_command(self, command):
        """Main command execution logic"""
        if command.command_type == CommandType.NOT_A_REQUEST:
            print("Not a request. No action taken.")
            shared_state.update_command("Idle")
            emit_status_update()
            return
        
        # Update the current command in the shared state
        current_command = str(command)
        shared_state.update_command(current_command)
        emit_status_update()

        print("Before command execution:")
        self.tool_inventory.print_inventory()

        if command.command_type == CommandType.DISPENSE:
            self._handle_dispense_command(command)
        elif command.command_type == CommandType.RETURN:
            self._handle_return_command()

        if command.command_type in [CommandType.DISPENSE, CommandType.RETURN]:
            self.last_command_type = command.command_type

        print("After command execution:")
        self.tool_inventory.print_inventory()
        
        # Update shared inventory state
        shared_state.update_inventory(self.tool_inventory)
        emit_status_update()
        
        # Reset the command display after processing
        time.sleep(2)  # Keep the command visible for a moment
        shared_state.update_command("Idle")
        emit_status_update()
    
    def _handle_dispense_command(self, command):
        """Handle tool dispensing logic"""
        if self.tool_inventory.dispense_tool(command.tool):
            location = ToolLocations.get_location(command)
            print(f"Dispensing {command.tool.value} from {location}")
            self.serial_communicator.send(location)
        else:
            print(f"Error: Tool {command.tool.value} is already dispensed.")
    
    def _handle_return_command(self):
        """Handle tool return logic"""
        # Take a photo to identify the tool
        self.camera_manager.take_photo()
        
        try:
            # Try to classify the tool from the photo
            classified_tool = self._classify_and_get_return_tool()
            if classified_tool:
                # Create a return command with the classified tool
                return_command = Command(f"{classified_tool.value} {CommandType.RETURN.value}")
                location = ToolLocations.get_location(return_command)
                self.serial_communicator.send(location)
                print(f"Tool {classified_tool.value} successfully returned.")
        except Exception as e:
            print(f"Error handling return: {e}")
    
    def _classify_and_get_return_tool(self):
        """Classify the tool in photo and determine which tool to return"""
        try:
            # Try to classify the tool from the photo
            classified_tool = self.tool_classifier.classify_tool_from_photo(PHOTO_PATH)
            print(f"Classified tool: {classified_tool.value}")
            
            # If the classified tool is marked as out, return it
            if not self.tool_inventory.is_tool_in(classified_tool):
                self.tool_inventory.return_tool(classified_tool)
                return classified_tool
            
            print(f"Warning: Classified tool {classified_tool.value} is already marked as in.")
            
            # Find any tool that's marked as out
            for tool in Tool:
                if not self.tool_inventory.is_tool_in(tool):
                    print(f"Using {tool.value} instead since it's marked as out.")
                    self.tool_inventory.return_tool(tool)
                    return tool
                    
            print("No tools are marked as out. Cannot return anything.")
            return None
            
        except Exception as e:
            print(f"Tool classification failed: {e}")
            # Even if classification fails, try to return any tool that's out
            for tool in Tool:
                if not self.tool_inventory.is_tool_in(tool):
                    print(f"Classification failed but found tool marked as out: {tool.value}")
                    self.tool_inventory.return_tool(tool)
                    return tool
            
            print("Classification failed and no tools are marked as out. Cannot proceed.")
            return None
    
    def _check_web_commands(self):
        """Check for commands from the web interface"""
        while True:
            command = get_command()
            if command:
                print(f"Executing command from web interface: {command}")
                self.execute_command(command)
            time.sleep(0.1)
    def get_last_command_type(self):
        return self.last_command_type
                            
    def start(self):
        """Start the control service"""
        print("Initializing tool classifier...")
        self.tool_classifier.initialize()
        
        print("Initializing serial communicator...")
        self.serial_communicator.initialize()
        
        # Start the web server in a separate thread
        print("Starting web server...")
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        web_thread.start()
        
        # Start the command checking thread
        print("Starting web command listener...")
        command_thread = threading.Thread(target=self._check_web_commands, daemon=True)
        command_thread.start()
        
        print("Starting voice transcription service...")
        self.voice_service.start()