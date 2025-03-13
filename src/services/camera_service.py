import cv2
import subprocess
import time
import os
import numpy
from config import PHOTO_PATH

class CameraManager:
    def __init__(self):
        self.photo_path = PHOTO_PATH # Default photo path
        self.device = "/dev/video0"  # Default video device

    def initialize(self):
        # First check if the camera is available
        if not os.path.exists(self.device):
            print(f"Error: Camera device {self.device} not found")
            return False
                
        # Configure camera using v4l2-ctl with the correct controls for your camera
        try:
            # Set resolution to 1280x720
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-fmt-video=width=1280,height=720,pixelformat=MJPG"], check=True)
            
            # Set brightness lower to avoid overexposure
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=brightness=20"], check=True)
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=contrast=55"], check=True)
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=saturation=64"], check=True)
            
            # Use manual exposure with the higher value that worked well in testing
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=auto_exposure=1"], check=True)  # Manual mode
            subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=exposure_time_absolute=750"], check=True)  # Higher exposure based on your testing
            
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
        
        # Simple capture without post-processing
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not capture frame")
            self.cap.release()
            self.cap = None
            return False
        
        # Print frame dimensions to confirm resolution
        height, width = frame.shape[:2]
        print(f"Captured frame with dimensions: {width}x{height}")
        
        # Save the raw frame without any processing
        cv2.imwrite(self.photo_path, frame)
        print(f"Photo taken and saved to {self.photo_path}")
        self.cap.release()
        self.cap = None
        return True
    def test_exposure_settings(self):
        """Test different exposure settings to find the ideal one"""
        # Use the current directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different exposure values
        for exposure in [100, 150, 200, 250, 300, 500, 750, 1000]:
            try:
                # Set manual exposure
                subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=auto_exposure=1"], check=True)
                subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl=exposure_time_absolute=" + str(exposure)], check=True)
                
                # Initialize camera
                self.cap = cv2.VideoCapture(0)
                time.sleep(2)
                
                # Discard initial frames
                for _ in range(3):
                    self.cap.read()
                    
                # Take photo
                ret, frame = self.cap.read()
                if ret:
                    filename = f"{script_dir}/exposure_{exposure}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved test image with exposure {exposure} to {filename}")
                    
                self.cap.release()
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error testing exposure {exposure}: {e}")
        
        print(f"Test images saved to {script_dir}")
        return True

if __name__ == "__main__":
    cm = CameraManager()
    # cm.get_camera_controls()
    cm.take_photo()
    # cm.test_exposure_settings()