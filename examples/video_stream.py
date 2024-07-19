import os
import sys

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa
from emotion_processor.main import EmotionRecognitionSystem

process = EmotionRecognitionSystem()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        process.video_stream_processing(frame)
        cv2.imshow('Emotion Recognition', frame)
        t = cv2.waitKey(5)
        if t == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
