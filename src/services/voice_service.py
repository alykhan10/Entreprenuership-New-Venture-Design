import os
import string
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from whisper_live.client import TranscriptionClient
from models.commands import Command
from config import CommandType, Tool

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'libs', 'WhisperLive'))

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

    def classify_text(self, transcribed_text):
        transcribed_text = transcribed_text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
        print(f"Transcribed text: {transcribed_text}")
        
        prompt = f"""
        Prompt:
        You are an AI that extracts voice commands from potentially inaccurate speech-to-text transcriptions. Your task is to interpret whether the user is asking the robot to dispense or return a tool, even if the transcription contains errors.

        ### Command Requirements:
        1. Look for words that sound similar to **"robot"** such as "row bot", "roh bot", etc.
        2. For **dispense** commands, identify words like "dispense", "give", "hand", "pass", etc.
        3. For **return** commands, identify words like "return", "take", "put back", etc.
        4. For tool names, be flexible with how they might be transcribed:
           - "straight mayo scissor" might be "straight may scissors", "straight maze scissor", etc.
           - "curved mayo scissor" might be "curve mayo", "curved may", etc. 
           - "scalpel" might be "scalp", "skull pole", etc.
           - "dissection clamp" might be "dissection lamp", "section clamp", etc.

        ### Response Format:
        - For a dispense command with an identified tool, return: **`"<tool> dispense"`**
        - For a return command (any tool can be returned), return: **`"not_a_request return"`**
        - If no valid command is found, return exactly: **`"not a request"`**

        Process the following transcription:
        "{transcribed_text}"
        """
        try:
            response = self.model.generate_content(prompt)
            result = response.text.replace("`", "").replace("'","").replace('"',"").strip().lower()
            print(f"LLM response: {result}")
            
            # Create a Command object from the response
            return Command(result)
        except Exception as e:
            print(f"Error during API call: {e}")
            return Command("not a request")

class VoiceTranscriptionService:
    def __init__(self, callback):
        self.client = None
        self.callback = callback
        self.last_text = ""
        
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
            return True
        except Exception as e:
            print(f"Error initializing transcription service: {e}")
            return False
            
    def _handle_transcription(self, text, is_final):
        if is_final and text and text != self.last_text:
            self.last_text = text
            if self.client and hasattr(self.client, 'client'):
                self.client.client.paused = True
            
            # Process the latest text segment
            if text and len(text) > 0:
                self.callback(text[-1])
                
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