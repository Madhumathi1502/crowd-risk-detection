import cv2

class CameraStream:
    def __init__(self, source=0):
        """
        Initialize the camera stream.
        source can be an integer (e.g., 0 for default webcam, 1 for DroidCam virtual camera)
        or a string (e.g., 'http://192.168.0.100:4747/video' for DroidCam IP camera).
        """
        self.source = source
        self.cap = cv2.VideoCapture(self.source)

    def is_opened(self):
        return self.cap.isOpened()

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
