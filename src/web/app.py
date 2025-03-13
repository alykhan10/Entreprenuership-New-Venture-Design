from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import os
import sys
import threading
import time

# Add parent directory to path to import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tools import ToolInventory
from config import Tool, CommandType
from models.commands import Command

app = Flask(__name__)
app.config['SECRET_KEY'] = 'o-robot-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Shared state between web app and control service
class SharedState:
    def __init__(self):
        self.tool_inventory = ToolInventory()
        self.current_command = "Idle"
        self.current_transcript = ""
        self.lock = threading.Lock()
    
    def update_inventory(self, inventory):
        with self.lock:
            self.tool_inventory = inventory
            
    def update_command(self, command):
        with self.lock:
            self.current_command = command
            
    def update_transcript(self, transcript):
        with self.lock:
            self.current_transcript = transcript
            
    def get_inventory_status(self):
        with self.lock:
            return {tool.value: not self.tool_inventory.is_tool_in(tool) for tool in Tool}
            
    def get_command(self):
        with self.lock:
            return self.current_command
            
    def get_transcript(self):
        with self.lock:
            return self.current_transcript

# Create shared state
shared_state = SharedState()

# Command queue for web app to control service communication
command_queue = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    inventory_status = shared_state.get_inventory_status()
    tools_out = sum(1 for status in inventory_status.values() if status)
    return jsonify({
        'tools_out': tools_out,
        'inventory': inventory_status,
        'current_command': shared_state.get_command(),
        'current_transcript': shared_state.get_transcript()
    })

@app.route('/api/dispense/<tool_name>', methods=['POST'])
def dispense_tool(tool_name):
    try:
        tool = Tool(tool_name)
        command_queue.append(Command(f"{tool.value} dispense"))
        return jsonify({'status': 'success', 'message': f'Command to dispense {tool_name} queued'})
    except (ValueError, KeyError) as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/return', methods=['POST'])
def return_tool():
    command_queue.append(Command("not_a_request return"))
    return jsonify({'status': 'success', 'message': 'Return command queued'})

def get_command():
    """Get next command from the queue if available"""
    if command_queue:
        return command_queue.pop(0)
    return None

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('status_update', {
        'inventory': shared_state.get_inventory_status(),
        'current_command': shared_state.get_command(),
        'current_transcript': shared_state.get_transcript(),
        'tools_out': sum(1 for status in shared_state.get_inventory_status().values() if status)
    })

def emit_status_update():
    """Emit status update to all clients"""
    inventory_status = shared_state.get_inventory_status()
    socketio.emit('status_update', {
        'inventory': inventory_status,
        'current_command': shared_state.get_command(),
        'current_transcript': shared_state.get_transcript(),
        'tools_out': sum(1 for status in inventory_status.values() if status)
    })

def run_web_server(debug=False, use_reloader=False):
    socketio.run(app, host='0.0.0.0', port=5000, debug=debug, use_reloader=use_reloader)

if __name__ == '__main__':
    run_web_server(debug=True)