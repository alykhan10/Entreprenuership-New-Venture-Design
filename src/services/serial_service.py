import serial
import time
from config import SERIAL_PORT, SERIAL_BAUD

class SerialCommunicator:
    def __init__(self):
        self.port = SERIAL_PORT
        self.baudrate = SERIAL_BAUD
        self.serial = None
        
    def initialize(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Serial connection established on {self.port} at {self.baudrate} baud")
            time.sleep(2)  # Give the serial connection time to stabilize
            return True
        except Exception as e:
            print(f"Failed to initialize serial connection: {e}")
            return False
            
    def send(self, command):
        if not self.serial:
            print("Serial not initialized")
            return False
            
        try:
            print(f"Sending command: {command}")
            self.serial.write(f"{command}\n".encode())
            time.sleep(0.1)  # Small delay to ensure command is sent
            return True
        except Exception as e:
            print(f"Serial communication error: {e}")
            return False
            
    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Serial connection closed")