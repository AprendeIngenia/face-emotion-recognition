import cv2
import numpy as np


class EmotionsVisualization:
    def __init__(self):
        self.emotion_colors = {
            'surprise': (184, 183, 83),
            'angry': (35, 50, 220),
            'disgust': (79, 164, 36),
            'sad': (186, 119, 4),
            'happy': (27, 151, 239),
            'fear': (128, 37, 146)
        }

    def main(self, emotions: dict, original_image: np.ndarray):
        for i, (emotion, score) in enumerate(emotions.items()):
            cv2.putText(original_image, emotion, (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.emotion_colors[emotion], 1,
                        cv2.LINE_AA)
            cv2.rectangle(original_image, (150, 15 + i * 40), (150 + int(score * 2.5), 35 + i * 40), self.emotion_colors[emotion],
                          -1)
            cv2.rectangle(original_image, (150, 15 + i * 40), (400, 35 + i * 40), (255, 255, 255), 1)

        return original_image
