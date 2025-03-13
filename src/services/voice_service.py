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

    def classify_text(self, transcribed_text, last_cmd_type=None):
        # Clean the input text
        transcribed_text = transcribed_text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
        print(f"Transcribed text: {transcribed_text}")

        # Only prioritize a RETURN command if the last command wasn’t already RETURN
        if last_cmd_type != CommandType.RETURN and "return" in transcribed_text and "robot" in transcribed_text:
            print("Detected return command with priority")
            return Command("not_a_request return")

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

        ### Important Rules:
        - MOST IMPORTANT: If you see both a dispense AND return command, prioritize the return command
        - If multiple tools are mentioned, focus on identifying the first complete command
        - For comma-separated phrases, treat each part as a potential separate command
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
            
            validated_command = self._validate_llm_response(llm_response, last_cmd_type)
            return validated_command
            
        except Exception as e:
            print(f"Error classifying command: {e}")
            return Command("not a request")
    
    
    def _validate_llm_response(self, llm_response, last_cmd_type=None):
        """Validate and fix the LLM response to ensure it contains valid tools and commands"""
        
        # Clean the response of any potential formatting or extra characters
        llm_response = llm_response.strip().lower()
        
        # First, check if both command types are present in the original text
        has_return = "return" in self.last_processed_text.lower() and ("robot" in self.last_processed_text.lower() or "tool" in self.last_processed_text.lower())
        dispense_keywords = ["scissor", "scalpel", "clamp", "mayo"]
        has_dispense = any(dk in self.last_processed_text.lower() for dk in dispense_keywords) and "robot" in self.last_processed_text.lower()
        
        # If we have both command types, prioritize based on last command
        if has_return and has_dispense:
            if last_cmd_type == CommandType.RETURN:
                # Last command was RETURN, so prioritize DISPENSE
                print("Last command was RETURN - overriding to prioritize DISPENSE")
                # Check if LLM found a valid dispense command
                if "dispense" in llm_response and any(tool in llm_response for tool in self.valid_tools) or any(tool in llm_response for tool in self.tool_variations):
                    # Continue with normal validation for dispense
                    pass
                else:
                    # Force a check for dispense keywords
                    for tool_var in self.tool_variations:
                        if tool_var in self.last_processed_text.lower():
                            print(f"Found tool in text: {tool_var}, creating dispense command")
                            valid_tool = self.tool_variations[tool_var]
                            return Command(f"{valid_tool} dispense")
            else:
                # Last command was DISPENSE or None, prioritize RETURN
                if has_return:
                    print("Last command was DISPENSE or None - overriding to prioritize RETURN")
                    return Command("not_a_request return")
        
        if llm_response == "not a request":
            # Double-check for potential missed return commands
            if has_return and last_cmd_type != CommandType.RETURN:
                print("Response was 'not a request' but text contains 'robot' and 'return' - overriding to return command")
                return Command("not_a_request return")
            # Double-check for potential missed dispense commands when last command was RETURN
            elif has_dispense and last_cmd_type == CommandType.RETURN:
                for tool_var in self.tool_variations:
                    if tool_var in self.last_processed_text.lower():
                        print(f"Response was 'not a request' but found tool {tool_var} after RETURN - overriding to dispense command")
                        valid_tool = self.tool_variations[tool_var]
                        return Command(f"{valid_tool} dispense")
            return Command("not a request")
        
        if "return" in llm_response:
            if last_cmd_type == CommandType.RETURN and has_dispense:
                # Override to look for dispense if last command was return
                print("LLM suggested RETURN but last command was already RETURN - checking for dispense instead")
                for tool_var in self.tool_variations:
                    if tool_var in self.last_processed_text.lower():
                        print(f"Found tool in text: {tool_var}, creating dispense command")
                        valid_tool = self.tool_variations[tool_var]
                        return Command(f"{valid_tool} dispense")
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
        self.last_command_time = 0
        self.command_cooldown = 3.0
        self.last_processed_text = ""  
        self.current_transcription = "" 
        
    def initialize(self):
        try:
            print("Initializing transcription client...")
            self.client = TranscriptionClient(
                host="localhost",
                port=9090,
                lang="en",
                translate=False,
                model="tiny.en",
                use_vad=True,
                callback=self._handle_transcription
            )
            return True
        except Exception as e:
            print(f"Error initializing transcription service: {e}")
            return False
            
    def _handle_transcription(self, text, is_final):
        current_time = time.time()
        
        if isinstance(text, list):
            if not text:  
                return
            # Just use the last segment of text, which is the most recent
            text = text[-1]
        
        # Only process final transcriptions
        if not is_final or not text:
            return
            
        # Extract only the new part of the transcription
        if text.startswith(self.current_transcription) and text != self.current_transcription:
            # Only process the new part that was added
            new_text = text[len(self.current_transcription):].strip()
            if new_text:
                print(f"Extracted new part of transcription: '{new_text}'")
                latest_text = new_text
            else:
                # No new content, probably just minor changes
                return
        else:
            # Complete replacement or unrelated text
            latest_text = text
            
        # Update our current transcription 
        self.current_transcription = text
        
        # Check for separate commands by looking for "robot" keyword
        # This finds positions of all occurrences of "robot" in the text
        robot_positions = self._find_all_occurrences(latest_text.lower(), "robot")
        
        if len(robot_positions) > 1:
            commands = []
            for i in range(len(robot_positions)):
                start_pos = robot_positions[i]
                if i < len(robot_positions) - 1:
                    end_pos = robot_positions[i + 1]
                    cmd = latest_text[start_pos:end_pos].strip()
                else:
                    cmd = latest_text[start_pos:].strip()
                commands.append(cmd)
            
            # Identify if we have both a return command and a dispense request
            has_return = any("return" in c.lower() for c in commands)
            dispense_keywords = ["scissor", "scalpel", "clamp", "mayo"]
            has_dispense = any(any(dk in c.lower() for dk in dispense_keywords) for c in commands)

            # Prioritize the opposite of the last command type, if both exist
            if has_return and has_dispense:
                last_cmd_type = (self.control_service.get_last_command_type()
                                 if self.control_service else None)
                if last_cmd_type == CommandType.RETURN:
                    print("Last command was RETURN - prioritizing DISPENSE this time")
                    for cmd in commands:
                        if any(dk in cmd.lower() for dk in dispense_keywords):
                            latest_text = cmd
                            break
                else:
                    print("Last command was DISPENSE or None - prioritizing RETURN this time")
                    for cmd in commands:
                        if "return" in cmd.lower():
                            latest_text = cmd
                            break
            elif has_return:
                print("Found return command - default prioritizing RETURN")
                for cmd in commands:
                    if "return" in cmd.lower():
                        latest_text = cmd
                        break
        
        # Skip if this is too similar to the last processed text
        if self._is_duplicate(latest_text, self.last_processed_text):
            print(f"Skipping duplicate command: '{latest_text}'")
            return

        if current_time - self.last_command_time < self.command_cooldown:
            print(f"Command rejected - cooldown period ({self.command_cooldown}s) not elapsed")
            return

        print(f"\n======= PROCESSING NEW COMMAND: '{latest_text}' =======\n")
        self.last_command_time = current_time
        self.last_processed_text = latest_text

        self.callback(latest_text)
        self.current_transcription = ""
        self.last_buffer_reset = current_time

    def _find_all_occurrences(self, text, substring):
        """Find all starting positions of substring in text"""
        positions = []
        pos = text.find(substring)
        while pos != -1:
            positions.append(pos)
            pos = text.find(substring, pos + 1)
        return positions
    
    def _is_duplicate(self, text1, text2):
        """Check if text1 is likely a duplicate of text2 using simple similarity metric"""
        # If either is empty, not a duplicate
        if not text1 or not text2:
            return False
            
        # Clean and normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match is duplicate
        if text1 == text2:
            return True
            
        # For simple similarity, check percentage of common words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
            
        # Calculate Jaccard similarity: intersection over union
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        
        return similarity > 0.8  # 80% similarity threshold
    
    def start(self):
        if not self.client:
            if not self.initialize():
                return False
                
        try:
            print("Starting transcription client...")
            self.client()  
            print("Transcription client started and running")
            return True
        except Exception as e:
            print(f"Error starting transcription service: {e}")
            import traceback
            traceback.print_exc()
            return False