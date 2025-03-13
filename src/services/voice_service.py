import os
import string
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from models.commands import Command
from config import CommandType, Tool
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'libs', 'WhisperLive'))

from whisper_live.client import TranscriptionClient

class EnvironmentLoader:
    @staticmethod
    def load():
        load_dotenv()  
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            return None
        return gemini_key

class CommandClassifier:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
        
        # Valid tools and commands mapping for validation
        self.valid_tools = {tool.value: tool.value for tool in Tool}
        # Add some common variations for more robust matching
        self.tool_variations = {
            "straight mayo scissor": Tool.STRAIGHT_MAYO_SCISSOR.value,
            "straight scissors": Tool.STRAIGHT_MAYO_SCISSOR.value,
            "curved mayo scissor": Tool.CURVED_MAYO_SCISSOR.value,
            "curved scissors": Tool.CURVED_MAYO_SCISSOR.value,
            "scalpel": Tool.SCALPEL.value,
            "dissection clamp": Tool.DISSECTION_CLAMP.value,
            "section clamp": Tool.DISSECTION_CLAMP.value,
            "clamp": Tool.DISSECTION_CLAMP.value
        }

    def classify_text(self, transcribed_text):
        
        
        # Clean the input text
        transcribed_text = transcribed_text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
        print(f"Transcribed text: {transcribed_text}")

        # Store the last processed text for better return command detection
        self.last_processed_text = transcribed_text    
        
        prompt = f"""
        Prompt:
        You are an AI that extracts voice commands from potentially inaccurate speech-to-text transcriptions. Your task is to interpret whether the user is asking the robot to dispense or return a tool, even if the transcription contains errors.

        ### Command Requirements:
        1. Look for words that sound similar to **"robot"** such as "row bot", "roh bot", "about", "block" etc.
        2. For **dispense** commands, identify words like "dispense", "give", "hand", "pass", "spend", etc.
        3. For **return** commands, identify words like "return", "take", "put back", etc.
        4. For tool names, be flexible with how they might be transcribed, but ONLY respond with EXACTLY one of these specific tool names:
        - straight_mayo_scissor
        - curved_mayo_scissor
        - scalpel
        - dissection_clamp

        For example:
        - dissection clamp may be transcribed as section clamp, clamp, etc. You should still recognize it as a dissection clamp.
        - curved mayo scissors may be transcibed as curved meiosis, curved scissors, etc. You should still recognize it as a curved mayo scissor.
        - straight mayo scissors may be transcribed as straight meiosis, straight scissors, etc. You should still recognize it as a straight mayo scissor.

        ### Important Rules:
        - If there are multiple commands or repetitions, just focus on identifying the FIRST complete command
        - For comma-separated phrases, treat each part as a potential separate command
        - Even if the command is repeated, like "robot return, robot return", still extract it
        - A single word "return" after "robot" is sufficient for a return command
        - Be extremely flexible with recognizing "robot return" variations

        ### Response Format:
        - For a dispense command with an identified tool, return: **`<exact_tool_name> dispense`**
        - For a return command (any tool can be returned), return: **`not_a_request return`**
        - If no valid command is found, return exactly: **`not a request`**

        Process the following transcription:
        "{transcribed_text}"
        """
        try:
            response = self.model.generate_content(prompt)
            # Clean the LLM response by removing any markdown formatting, quotes, or backticks
            llm_response = response.text.strip()
            llm_response = llm_response.replace('```', '').replace('`', '').strip()
            print(f"LLM response: {llm_response}")
            
            # Validate the LLM response
            validated_command = self._validate_llm_response(llm_response)
            return validated_command
            
        except Exception as e:
            print(f"Error classifying command: {e}")
            return Command("not a request")
    
    def _validate_llm_response(self, llm_response):
        """Validate and fix the LLM response to ensure it contains valid tools and commands"""
        
        # Clean the response of any potential formatting or extra characters
        llm_response = llm_response.strip().lower()
        
        # If response is "not a request"
        if llm_response == "not a request":
            # Double-check for potential missed return commands
            if "return" in self.last_processed_text and "robot" in self.last_processed_text:
                print("Response was 'not a request' but text contains 'robot' and 'return' - overriding to return command")
                return Command("not_a_request return")
            return Command("not a request")
        
        # Check for return command 
        if "return" in llm_response:
            return Command("not_a_request return")
        
        # For dispense commands, ensure proper format and valid tool
        try:
            parts = llm_response.split()
            if len(parts) < 2:
                print(f"Invalid command format. Defaulting to 'not a request'")
                return Command("not a request")
            
            # Extract command type (usually the last word)
            command_type = parts[-1].lower()
            if command_type != "dispense":
                print(f"Invalid command type '{command_type}'. Defaulting to 'not a request'")
                return Command("not a request")
            
            # Extract tool name (everything except the last word)
            tool = " ".join(parts[:-1]).lower()
            
            # Validate tool name
            if tool in self.valid_tools:
                # Direct match with enum values
                valid_tool = self.valid_tools[tool]
            elif tool in self.tool_variations:
                # Match with variation
                valid_tool = self.tool_variations[tool]
            else:
                print(f"Invalid tool name '{tool}'. Defaulting to 'not a request'")
                return Command("not a request")
            
            # Create valid command string
            valid_command = f"{valid_tool} dispense"
            print(f"Validated command: {valid_command}")
            return Command(valid_command)
            
        except Exception as e:
            print(f"Error during validation: {e}")
            return Command("not a request")

class VoiceTranscriptionService:
    def __init__(self, callback):
        self.client = None
        self.callback = callback
        self.last_text = ""
        self.last_command_time = 0
        self.command_cooldown = 3.0
        self.buffer_reset_interval = 5.0  # Reset buffer every 5 seconds
        self.last_buffer_reset = time.time()
        
    def initialize(self):
        try:
            self.client = TranscriptionClient(
                host="localhost",
                port=9090,
                lang="en",
                translate=False,
                model="tiny.en",
                use_vad=True,
                callback=self._handle_transcription
            )
            self.client.last_process_time = 0
            
            # Override the TranscriptionClient's record method to implement periodic buffer clearing
            original_record = self.client.record
            
            def record_with_buffer_reset(*args, **kwargs):
                # Periodically reset buffer
                self.client.frames = b""
                return original_record(*args, **kwargs)
                
            self.client.record = record_with_buffer_reset
            
            return True
        except Exception as e:
            print(f"Error initializing transcription service: {e}")
            return False
            
    def _handle_transcription(self, text, is_final):
        current_time = time.time()
        
        # Periodically reset buffer regardless of commands
        if current_time - self.last_buffer_reset > self.buffer_reset_interval:
            if hasattr(self.client, 'reset_buffer'):
                self.client.reset_buffer()
                print("[AUTO] Buffer reset due to interval")
            self.last_buffer_reset = current_time
        
        if is_final and text and text != self.last_text:
            # Only process the latest segment from the array
            latest_text = text[-1] if isinstance(text, list) and text else text
            
            # Reset the internal state to prevent contamination
            self.last_text = ""  # Reset last text instead of storing it
            
            # Check if enough time has passed since the last command
            if current_time - self.last_command_time < self.command_cooldown:
                print(f"Command rejected - cooldown period ({self.command_cooldown}s) not elapsed")
                # Still reset buffer even when rejecting command
                if hasattr(self.client, 'reset_buffer'):
                    self.client.reset_buffer()
                    print("[COOLDOWN] Buffer reset when rejecting command")
                return
                
            if self.client and hasattr(self.client, 'client'):
                self.client.client.paused = True
            
            # Process the command
            print(f"\n======= PROCESSING NEW COMMAND: '{latest_text}' =======\n")
            self.last_command_time = current_time
            self.callback(latest_text)
            
            # Multiple aggressive buffer clearing approaches
            if hasattr(self.client, 'reset_buffer'):
                self.client.reset_buffer()
                
            # Also try to reset parent client's buffer if available
            if hasattr(self.client, 'client') and hasattr(self.client.client, 'close_websocket'):
                # Force socket reconnection to clear server-side buffers
                try:
                    # Store connection info
                    host = self.client.client.host
                    port = self.client.client.port
                    
                    # Close and reconnect
                    self.client.client.close_websocket()
                    time.sleep(0.5)  # Short delay to ensure disconnection
                    
                    # Reconnect with a new websocket
                    self.client.client.get_client_socket()
                    print("[RECONNECT] WebSocket reconnected to clear server-side buffers")
                except Exception as e:
                    print(f"Error during websocket reconnection: {e}")
                    
            # Force garbage collection
            import gc
            gc.collect()
                    
            if self.client and hasattr(self.client, 'client'):
                self.client.client.paused = False
    
    def start(self):
        if not self.client:
            if not self.initialize():
                return False
                
        try:
            self.client()  # Start the transcription client
            return True
        except Exception as e:
            print(f"Error starting transcription service: {e}")
            return False
    def __init__(self, callback):
        self.client = None
        self.callback = callback
        self.last_text = ""
        self.last_command_time = 0
        self.command_cooldown = 3.0
        
    def initialize(self):
        try:
            self.client = TranscriptionClient(
                host="localhost",
                port=9090,
                lang="en",
                translate=False,
                model="tiny.en",
                use_vad=True,
                callback=self._handle_transcription
            )
            self.client.last_process_time = 0
            return True
        except Exception as e:
            print(f"Error initializing transcription service: {e}")
            return False
        
            
    def _handle_transcription(self, text, is_final):
        current_time = time.time()
        
        if is_final and text and text != self.last_text:
            self.last_text = text
            
            # Check if enough time has passed since the last command
            if current_time - self.last_command_time < self.command_cooldown:
                print(f"Command rejected - cooldown period ({self.command_cooldown}s) not elapsed")
                return
                
            if self.client and hasattr(self.client, 'client'):
                self.client.client.paused = True
            
            # Process the latest text segment
            if text and len(text) > 0:
                print(f"\n======= PROCESSING NEW COMMAND: '{text[-1]}' =======\n")
                self.last_command_time = current_time
                self.callback(text[-1])
                
                # Aggressive reset:
                if hasattr(self.client, 'reset_buffer'):
                    self.client.reset_buffer()
                    
                # Force garbage collection to clean up any lingering references
                import gc
                gc.collect()
                    
            if self.client and hasattr(self.client, 'client'):
                self.client.client.paused = False
    
    def start(self):
        if not self.client:
            if not self.initialize():
                return False
                
        try:
            self.client()  # Start the transcription client
            return True
        except Exception as e:
            print(f"Error starting transcription service: {e}")
            return False