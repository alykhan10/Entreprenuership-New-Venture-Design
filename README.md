# O-Robot Control Service

O-Robot is a control service designed to manage various robotic functionalities including tool dispensing, tool returning, voice command processing, and more. This README provides an overview of the components, classes, tools, and machine learning models used in the O-Robot project. This system is running on a Raspberry Pi 5 with a microphone, webcam, and serial port connected to the Pi.

## Table of Contents
- [Overview](#overview)
- [Components](#components)
- [Classes](#classes)
- [Tools](#tools)
- [Serial Interface](#serial-interface)
- [Machine Learning Model](#machine-learning-model)
- [Usage](#usage)
- [Launch Script](#launch-script)

## Overview
The O-Robot Control Service is a comprehensive system that integrates various services to control a robot. It includes components for voice transcription, tool classification, camera management, and serial communication. The system is designed to handle voice commands to dispense and return tools, and it uses machine learning models to classify tools from images. Additionally, we integrate with the Gemini API to classify text and extract sentiment. The system also includes a web app to view the state of the system, issue commands, and see the inventory state.

## Components
The main components of the O-Robot Control Service are:
- **Voice Service**: Handles voice transcription and command classification.
- **Camera Service**: Manages the camera and captures images for tool classification.
- **Serial Service**: Manages serial communication with the robot.
- **Tool Classifier**: Uses a machine learning model to classify tools from images.
- **Control Service**: Orchestrates the overall functionality and integrates all services.
- **Web App**: Provides a user interface to view system status, issue commands, and manage tool inventory.

## Classes
- **ControlService**: Located in `control_service.py`, this class is the main orchestrator of the O-Robot system. It integrates all other services and handles the execution of commands.
- **CameraManager**: Located in `camera_service.py`, this class manages the camera device, configures it, and captures images for tool classification.
- **SerialCommunicator**: Located in `serial_service.py`, this class handles serial communication with the robot. It initializes the serial connection, sends commands, and closes the connection.
- **VoiceTranscriptionService**: Located in `voice_service.py`, this class handles voice transcription using the WhisperLive library and processes transcribed text to classify commands.
- **ToolClassifier**: Located in `tool_classifier.py`, this class uses a machine learning model to classify tools from images. It preprocesses images and performs inference to identify tools.
- **ToolInventory**: Located in `tools.py`, this class manages the inventory of tools, tracking which tools are in and which are out.
- **Command**: Located in `commands.py`, this class represents a command with a tool and a command type (dispense, return, or not a request).
- **ToolLocations**: Located in `commands.py`, this class provides the locations for dispensing and returning tools.

## Tools
The tools used in the O-Robot system are defined in `config.py` and include:
- Straight Mayo Scissor
- Curved Mayo Scissor
- Scalpel
- Dissection Clamp

## Serial Interface
The serial interface is managed by the `SerialCommunicator` class. It communicates with the robot using a specified serial port and baud rate defined in `config.py`.

### Configuration
- **Serial Port**: `ttyUSB0`
- **Baud Rate**: `9600`

### Commands
Commands are sent to the robot to dispense or return tools. The specific locations for each tool are defined in `ToolLocations`.

## Machine Learning Model
The `ToolClassifier` class uses a machine learning model to classify tools from images. The model is a ResNet-50 architecture, trained on a custom dataset of tool images captured using the camera service.

### Training Process
- **Data Collection**: Images of each tool were captured using the camera service and labeled accordingly.
- **Preprocessing**: Images were resized and normalized to ensure consistency.
- **Model Training**: The ResNet-50 model was trained using the labeled dataset. The training process involved fine-tuning the model to achieve high accuracy in tool classification.
- **Model Deployment**: The trained model was saved and integrated into the `ToolClassifier` class for real-time inference.

## Usage
To start the O-Robot Control Service, run the `main.py` script. The service will initialize all components and start listening for voice commands and web interface commands.

The service will handle voice commands to dispense and return tools, capture images for tool classification, and communicate with the robot via the serial interface.

## Launch Script
The `launch.sh` script is used to start the O-Robot system, including the WhisperLive transcription server. This script sets up the environment, starts the transcription server, and then launches the main application.

### Steps in `launch.sh`:
1. **Set the base directory**: The script sets the base directory to the location of the script.
2. **Activate the virtual environment**: The script activates the Python virtual environment.
3. **Start the WhisperLive transcription server**: The script starts the WhisperLive server in the background and waits for it to initialize.
4. **Launch the main application**: The script starts the O-Robot Control Service.
5. **Clean up**: When the main application exits, the script stops the WhisperLive server and performs cleanup.

To use the `launch.sh` script, run the following command:
```sh
./launch.sh
```

This will start the entire O-Robot system, including the voice transcription server and the control service.