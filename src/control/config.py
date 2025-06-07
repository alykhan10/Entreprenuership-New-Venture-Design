import os
from enum import Enum

# File paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'libs', 'classify', 'tool_classifier_new.pth')
PHOTO_PATH = '/home/orobot/orobot/src/photo.jpg'

# Serial configuration
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 9600

# Tool classification configuration
CLASS_NAMES = ["curved-mayo-scissor", "dissection-clamp", "scalpel", "straight-mayo-scissor"]
TARGET_SIZE = (256, 144)

# Command and tool definitions
class CommandType(Enum):
    DISPENSE = "dispense"
    RETURN = "return"
    NOT_A_REQUEST = "not_a_request"

class Tool(Enum):
    STRAIGHT_MAYO_SCISSOR = "straight_mayo_scissor"
    CURVED_MAYO_SCISSOR = "curved_mayo_scissor" 
    SCALPEL = "scalpel"
    DISSECTION_CLAMP = "dissection_clamp"
