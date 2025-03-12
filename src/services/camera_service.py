import cv2
from config import PHOTO_PATH

class CameraManager:
    def __init__(self):
        self.photo_path = PHOTO_PATH
        
    def initialize(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        return True

    def take_photo(self):
        if not hasattr(self, 'cap') or self.cap is None:
            if not self.initialize():
                return False
                
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(self.photo_path, frame)
            print(f"Photo taken and saved to {self.photo_path}")
            self.cap.release()
            self.cap = None
            return True
        else:
            print("Error: Could not read frame from camera.")
            self.cap.release()
            self.cap = None
            return False