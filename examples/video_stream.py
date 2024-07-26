import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emotion_processor.main import EmotionRecognitionSystem
from camera import Camera


class VideoStream:
    def __init__(self, cam: Camera, emotion_recognition_system: EmotionRecognitionSystem):
        self.camera = cam
        self.emotion_recognition_system = emotion_recognition_system

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if ret:
                frame = self.emotion_recognition_system.frame_processing(frame)
                cv2.imshow('Emotion Recognition', frame)
                t = cv2.waitKey(5)
                if t == 27:
                    break

            else:
                Exception(f"No cam connected")
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = Camera(0, 1280, 720)
    emotion_recognition_system = EmotionRecognitionSystem()
    video_stream = VideoStream(camera, emotion_recognition_system)
    video_stream.run()


