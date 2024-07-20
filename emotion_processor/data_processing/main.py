from emotion_processor.data_processing.eyebrows_processing import EyeBrowsPointsProcessing
from emotion_processor.data_processing.eyes_processing import EyesPointsProcessing
from emotion_processor.data_processing.nose_processing import NosePointsProcessing
from emotion_processor.data_processing.mouth_processing import MouthPointsProcessing


class PointsProcessing:
    def __init__(self):
        self.eyes_brows = EyeBrowsPointsProcessing()
        self.eyes = EyesPointsProcessing()
        self.nose = NosePointsProcessing()
        self.mouth = MouthPointsProcessing()
        pass

    def main(self, eye_brows_points: dict, eyes_points: dict, nose_points: dict, mouth_points: dict):
        eyebrows_info = self.eyes_brows.main(eye_brows_points)
        self.eyes.main(eyes_points)
        self.nose.main(nose_points)
        self.mouth.main(mouth_points)
        print(f'\ninfo eyebrows: {eyebrows_info}')

