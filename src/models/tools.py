from config import Tool

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
    
    def print_inventory(self):
        print("=== TOOL INVENTORY STATUS ===")
        for tool in Tool:
            status = "IN" if self.inventory[tool] else "OUT"
            print(f"{tool.value}: {status}")
        print("===========================")