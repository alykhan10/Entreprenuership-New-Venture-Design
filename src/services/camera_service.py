import cv2
import subprocess
import time
import os
import numpy
from config import PHOTO_PATH

class CameraManager:
    def __init__(self):
        self.photo_path = PHOTO_PATH
        self.device = "/dev/video0"

    def initialize(self):
        # First check if the camera is available
        if not os.path.exists(self.device):
            print(f"Error: Camera device {self.device} not found")
            return False
                
        # Configure camera using v4l2-ctl with the correct controls
        try:
            # Set resolution to 1280x720
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-fmt-video=width=1280,height=720,pixelformat=MJPG"], check=True)
            
            # Set brightness lower to avoid overexposure
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=brightness=20"], check=True)
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=contrast=55"], check=True)
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=saturation=64"], check=True)
            
            # Use manual exposure 
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=auto_exposure=1"], check=True) 
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=exposure_time_absolute=750"], check=True)  
            
            # Additional balanced settings
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=white_balance_automatic=1"], check=True)
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=power_line_frequency=1"], check=True)  # 50Hz
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=backlight_compensation=15"], check=True)
                
        except subprocess.SubprocessError as e:
            print(f"Warning: Failed to configure camera with v4l2-ctl: {e}. Falling back to OpenCV.")
            
        # Now initialize OpenCV capture
        self.cap = cv2.VideoCapture(0)
        
        # Set OpenCV capture properties for resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
            
        return True

    def take_photo(self):
        if not hasattr(self, 'cap') or self.cap is None:
            if not self.initialize():
                return False
        
        # Give the camera time to adjust to new settings
        time.sleep(2)
        
        # Discard initial frames to allow camera to stabilize
        for _ in range(3):
            self.cap.read()
        
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not capture frame")
            self.cap.release()
            self.cap = None
            return False
        
        height, width = frame.shape[:2]
        
        cv2.imwrite(self.photo_path, frame)
        print(f"Photo taken and saved to {self.photo_path}")
        self.cap.release()
        self.cap = None
        return True
    
if __name__ == "__main__":
    cm = CameraManager()
    cm.take_photo()
