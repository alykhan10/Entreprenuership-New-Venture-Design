#!/bin/bash
# filepath: c:\Users\connor\Documents\school\nvd\orobot\launch.sh

# Display start message
echo "==========================="
echo "Starting O-Robot System"
echo "==========================="

# Set the base directory to the script's location
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Base directory: $BASE_DIR"

# Activate the virtual environment
echo "Activating virtual environment..."
source "$BASE_DIR/venv/bin/activate" 

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment!"
    exit 1
fi

# Launch the Whisper server in the background
echo "Starting Whisper Live transcription server..."
python "$BASE_DIR/src/libs/WhisperLive/run_server.py" --port 9090 --backend faster_whisper &

# Store the PID of the server
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for the server to initialize
echo "Waiting for server to initialize (5 seconds)..."
sleep 5

# Check if server is still running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "Server failed to start or crashed!"
    exit 1
fi

# Launch the main application
echo "Starting O-Robot Control Service..."
cd "$BASE_DIR/src"
python main.py

# When the main application exits, clean up
echo "Main application exited. Cleaning up..."
kill $SERVER_PID
echo "Whisper server stopped."
echo "O-Robot system shutdown complete."