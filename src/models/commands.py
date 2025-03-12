from config import CommandType, Tool

class Command:
    def __init__(self, command_string):
        if command_string == "not a request":
            self.tool = None
            self.command_type = CommandType.NOT_A_REQUEST
            return
        
        parts = command_string.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid command string: {command_string}")
        
        tool, command = parts[0], parts[1]
        
        # Special case for return - no need for a specific tool
        if command.strip() == CommandType.RETURN.value and tool.strip() == "not_a_request":
            self.tool = None
            self.command_type = CommandType.RETURN
            return
            
        self.tool = Tool(tool.strip())  # Ensure valid tool
        self.command_type = CommandType(command.strip())  # Ensure valid command type

    def __str__(self):
        # Safely handle None tool
        tool_value = self.tool.value if self.tool else "any tool"
        command_value = self.command_type.value if self.command_type else "no command"
        return f"{tool_value} {command_value}"

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
    def get_location(command):
        if command.command_type == CommandType.DISPENSE:
            return ToolLocations.DISPENSE_LOCATIONS[command.tool]
        elif command.command_type == CommandType.RETURN:
            return ToolLocations.RETURN_LOCATIONS[command.tool]
        else:
            raise ValueError(f"Invalid command type: {command.command_type}")