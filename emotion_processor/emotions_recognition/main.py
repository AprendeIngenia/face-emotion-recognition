from typing import Dict
from emotion_processor.emotions_recognition.features.emotion_score import EmotionScore
from .emotions.suprise_score import SurpriseScore
from .emotions.angry_score import AngryScore
from .emotions.disgust_score import DisgustScore
from .emotions.sad_score import SadScore
from .emotions.happy_score import HappyScore
from .emotions.fear_Score import FearScore


class EmotionRecognition:
    def __init__(self):
        self.emotions: Dict[str, EmotionScore] = {
            'surprise': SurpriseScore(),
            'angry': AngryScore(),
            'disgust': DisgustScore(),
            'sad': SadScore(),
            'happy': HappyScore(),
            'fear': FearScore(),
        }

    def recognize_emotion(self, processed_features: dict) -> str:
        scores = {emotion: scorer.calculate_score(processed_features) for emotion, scorer in self.emotions.items()}
        recognized_emotion = max(scores, key=scores.get)
        return recognized_emotion
