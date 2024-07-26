from abc import ABC, abstractmethod
from emotion_processor.emotions_recognition.features.emotion_score import EmotionScore
from emotion_processor.emotions_recognition.features.feature_implementation import (BasicEyebrowsCheck, BasicEyesCheck,
                                                                                    BasicNoseCheck, BasicMouthCheck)


class WeightedEmotionScore(EmotionScore, ABC):
    def __init__(self, eyebrows_weight, eyes_weight, nose_weight, mouth_weight):
        self.eyebrows_weight = eyebrows_weight
        self.eyes_weight = eyes_weight
        self.nose_weight = nose_weight
        self.mouth_weight = mouth_weight
        self.eyebrows_check = BasicEyebrowsCheck()
        self.eyes_check = BasicEyesCheck()
        self.nose_check = BasicNoseCheck()
        self.mouth_check = BasicMouthCheck()

    def calculate_score(self, features: dict) -> dict:
        eyebrows_result = self.eyebrows_check.check_eyebrows(features['eyebrows'])
        eyes_result = self.eyes_check.check_eyes(features['eyes'])
        nose_result = self.nose_check.check_nose(features['nose'])
        mouth_result = self.mouth_check.check_mouth(features['mouth'])

        eyebrows_score = self.calculate_eyebrows_score(eyebrows_result)
        eyes_score = self.calculate_eyes_score(eyes_result)
        nose_score = self.calculate_nose_score(nose_result)
        mouth_score = self.calculate_mouth_score(mouth_result)

        total_score = (eyebrows_score * self.eyebrows_weight +
                       eyes_score * self.eyes_weight +
                       nose_score * self.nose_weight +
                       mouth_score * self.mouth_weight)
        return {self.__class__.__name__.replace("Score", "").lower(): total_score}

    @abstractmethod
    def calculate_eyebrows_score(self, eyebrows_result: str) -> float:
        pass

    @abstractmethod
    def calculate_eyes_score(self, eyes_result: str) -> float:
        pass

    @abstractmethod
    def calculate_nose_score(self, nose_result: str) -> float:
        pass

    @abstractmethod
    def calculate_mouth_score(self, mouth_result: str) -> float:
        pass
