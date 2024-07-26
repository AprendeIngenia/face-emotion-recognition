from emotion_processor.emotions_recognition.features.weights_emotion_score import WeightedEmotionScore


class SadScore(WeightedEmotionScore):
    def __init__(self):
        super().__init__(eyebrows_weight=0.30, eyes_weight=0.30, nose_weight=0.1, mouth_weight=0.3)

    def calculate_eyebrows_score(self, eyebrows_result: str) -> float:
        score = 0.0
        if 'eyebrows together' in eyebrows_result:
            score += 60.0
        if 'right eyebrow: lowered' in eyebrows_result:
            score += 20.0
        if 'left eyebrow: lowered' in eyebrows_result:
            score += 20.0
        return score

    def calculate_eyes_score(self, eyes_result: str) -> float:
        if 'closed eyes' in eyes_result:
            return 100.0
        return 0.0

    def calculate_nose_score(self, nose_result: str) -> float:
        if 'neutral nose' in nose_result:
            return 100.0
        return 0.0

    def calculate_mouth_score(self, mouth_result: str) -> float:
        score = 0.0
        if 'closed mouth' in mouth_result:
            score += 30.0
        if 'no right smile' in mouth_result:
            score += 35.0
        if 'no left smile' in mouth_result:
            score += 35.0
        return score
