from emotion_processor.data_processing.feature_processor import FeatureProcessor
from emotion_processor.data_processing.nose.nose_processing import NosePointsProcessing


class NoseProcessor(FeatureProcessor):
    def __init__(self):
        self.processor = NosePointsProcessing()

    def process(self, points: dict):
        self.processor.main(points)