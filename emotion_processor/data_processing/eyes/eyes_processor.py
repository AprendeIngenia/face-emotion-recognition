from emotion_processor.data_processing.feature_processor import FeatureProcessor
from emotion_processor.data_processing.eyes.eyes_processing import EyesPointsProcessing


class EyesProcessor(FeatureProcessor):
    def __init__(self):
        self.processor = EyesPointsProcessing()

    def process(self, points: dict):
        self.processor.main(points)