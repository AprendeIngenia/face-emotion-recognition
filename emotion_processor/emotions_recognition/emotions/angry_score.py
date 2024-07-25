from emotion_processor.emotions_recognition.features.emotion_score import EmotionScore
from emotion_processor.emotions_recognition.features.feature_implementation import (BasicEyebrowsCheck, BasicEyesCheck,
                                                                                    BasicNoseCheck, BasicMouthCheck)


class AngryScore(EmotionScore):
    def __init__(self):
        self.eyebrows_check = BasicEyebrowsCheck()
        self.eyes_check = BasicEyesCheck()
        self.nose_check = BasicNoseCheck()
        self.mouth_check = BasicMouthCheck()

    def calculate_score(self, features: dict) -> float:
        score = 0.0
        eyebrows_result = self.eyebrows_check.check_eyebrows(features['eyebrows'])
        eyes_result = self.eyes_check.check_eyes(features['eyes'])
        nose_result = self.nose_check.check_nose(features['nose'])
        mouth_result = self.mouth_check.check_mouth(features['mouth'])

        # print(f"Eyebrows Check: {eyebrows_result}")
        # print(f"Eyes Check: {eyes_result}")
        # print(f"Nose Check: {nose_result}")
        # print(f"Mouth Check: {mouth_result}")
        return score
