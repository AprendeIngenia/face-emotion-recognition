from abc import ABC, abstractmethod


class EmotionScore(ABC):
    @abstractmethod
    def calculate_score(self, features: dict) -> float:
        pass
    