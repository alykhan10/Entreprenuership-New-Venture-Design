import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'libs', 'WhisperLive'))

import google.generativeai as genai
import serial
from whisper_live.client import TranscriptionClient
from dotenv import load_dotenv

class EnvironmentLoader:
    @staticmethod
    def load():
        load_dotenv()  # Load environment variables from .env file
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            print("Error: GOOGLE_API_KEY environment variable not set. Exiting.")
            exit()
        return gemini_key

class CommandClassifier:
    TOOLS = ["straight mayo scissor", "curved mayo scissor", "scalpel", "dissection clamp"]

    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')

    def classify(self, transcribed_text):
        prompt = f"""
            You are a highly skilled medical assistant. Your task is to analyze transcribed text and determine if it contains a direct request for one of the following tools: {', '.join(CommandClassifier.TOOLS)}.

            Commands issues to you will start with the command phrase "robot". While you should not ignore commands not containing this phrase, pay special attention to commands containing the word "robot".

            A direct request means the speaker is asking for one of the tools to be provided to them or is indicating that they need one of the tools to be given to them. A request is usually expressed using phrases like "get me", "hand me", "give me", "I need", "can you bring". 
            A mere mention of a tool does NOT constitute a request. You should ignore casual chatter or background conversation. Pay close attention to the intent of the speaker. 
            Note that the transcription client makes errors, you will have to do some inference as it is not 100% accurate.

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
            response = self.model.generate_content(prompt)
            result = response.text.strip().lower()
            if result in self.TOOLS:
                return result
            else:
                return "not a request"
        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error Processing Request"

class SerialCommunicator:
    DEVICE_NAME = "/dev/ttyUSB0"

    def __init__(self):
        self.ser = serial.Serial(self.DEVICE_NAME, 9600)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send(self, data):
        self.ser.write(data.encode('utf-8'))
        # debug
        line = self.ser.readline().decode('utf-8').rstrip()
        print(line)


class ControlService:
    def __init__(self):
        self.GEMINI_KEY = EnvironmentLoader.load()
        self.tool_classifier = CommandClassifier(self.GEMINI_KEY)
        self.serial_communicator = SerialCommunicator()

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

            if response != "not a request":
                self.serial_communicator.send(response)
            self.client.client.paused = False

    def start(self):
        # Start the transcription client
        self.client()

if __name__ == "__main__":
    control_service = ControlService()
    control_service.start()