from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore


class SurpriseScore(WeightedEmotionScore):
    def __init__(self):
        super().__init__(eyebrows_weight=0.40, eyes_weight=0.25, nose_weight=0.1, mouth_weight=0.25)

    def calculate_eyebrows_score(self, eyebrows_result: str) -> float:
        score = 0.0
        if 'eyebrows separated' in eyebrows_result:
            score += 10.0
        if 'right eyebrow: raised' in eyebrows_result:
            score += 45.0
        if 'left eyebrow: raised' in eyebrows_result:
            score += 45.0
        return score

    def calculate_eyes_score(self, eyes_result: str) -> float:
        if 'open eyes' in eyes_result:
            return 100.0
        return 0.0

    def calculate_nose_score(self, nose_result: str) -> float:
        if 'neutral nose' in nose_result:
            return 100.0
        return 0.0

    def calculate_mouth_score(self, mouth_result: str) -> float:
        score = 0.0
        if 'open mouth' in mouth_result:
            score += 80.0
        if 'no right smile' in mouth_result:
            score += 10.0
        if 'no left smile' in mouth_result:
            score += 10.0
        return score
