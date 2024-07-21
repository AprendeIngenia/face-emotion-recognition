from emotion_processor.data_processing.feature_processor import FeatureProcessor
from emotion_processor.data_processing.mouth.mouth_processing import MouthPointsProcessing


class MouthProcessor(FeatureProcessor):
    def __init__(self):
        self.processor = MouthPointsProcessing()

    def process(self, points: dict):
        self.processor.main(points)