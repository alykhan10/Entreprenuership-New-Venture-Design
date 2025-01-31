import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'libs', 'WhisperLive'))

import google.generativeai as genai
from whisper_live.client import TranscriptionClient
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

TOOLS = ["straight mayo scissor", "curved mayo scissor", "scalpel", "dissection clamp"]

class ControlService:
    def __init__(self):
        self.GEMINI_KEY = os.getenv("GOOGLE_API_KEY")  # Load from env variable
        if not self.GEMINI_KEY:
            print("Error: GOOGLE_API_KEY environment variable not set. Exiting.")
            exit()
        genai.configure(api_key=self.GEMINI_KEY) # Configure the API with the key
        
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
            # print(f"Transcribed Text: {text}")
            self.last_text = text
            self.client.client.paused = True

            # Send the transcribed text to the LLM
            response = self.classify_tool_request(text[-1])
            print(f"Transcribed Text: {text[-1]}\n LLM Response: {response}")
            
            self.client.client.paused = False
        # else:
            # os.system("cls" if os.name == "nt" else "clear")

    def classify_tool_request(self, transcribed_text):
        model = genai.GenerativeModel('gemini-1.5-flash-8b')  

        prompt = f"""
            You are a highly skilled medical assistant. Your task is to analyze transcribed text and determine if it contains a direct request for one of the following tools: {', '.join(TOOLS)}.

            Commands issues to you will start with the command phrase "robot". While you should not ignore commands not containing this phrase, pay special attention to commands containing the word "robot".

            A direct request means the speaker is asking for one of the tools to be provided to them or is indicating that they need one of the tools to be given to them. A request is usually expressed using phrases like "get me", "hand me", "give me", "I need", "can you bring". 
            A mere mention of a tool does NOT constitute a request. You should ignore casual chatter or background conversation. Pay close attention to the intent of the speaker. 

            Here are some examples of transcribed text and what you should output:
            Transcribed text: "I need the scalpel"
            Output: scalpel

            Transcribed text: "Can you hand me the straight mayo scissor?"
            Output: straight mayo scissor

            Transcribed text: "The scalpel is over there"
            Output: not a request

            Transcribed text: "I almost ran my hand through with a scalpel"
            Output: not a request

            If the text contains a direct request for a tool from the list, return just the name of the requested tool in lowercase.
            If there is no direct request for a tool, return "not a request".

            Here is the transcribed text:
            "{transcribed_text}"
        """

        try:
            response = model.generate_content(prompt)
            result = response.text.strip().lower()
            if result in TOOLS:
                return result
            else:
                return "not a request"
        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error Processing Request"


    def start(self):
        # Start the transcription client
        self.client()

# Example usage
if __name__ == "__main__":
    control_service = ControlService()
    control_service.start()