from emotion_processor.data_processing.feature_processor import FeatureProcessor
from emotion_processor.data_processing.eyebrows.eyebrows_processor import EyeBrowsProcessor
from emotion_processor.data_processing.eyes.eyes_processor import EyesProcessor
from emotion_processor.data_processing.nose.nose_processor import NoseProcessor
from emotion_processor.data_processing.mouth.mouth_processor import MouthProcessor


class PointsProcessing:
    def __init__(self):
        self.processors: dict[str, FeatureProcessor] = {
            'eyebrows': EyeBrowsProcessor(),
            'eyes': EyesProcessor(),
            'nose': NoseProcessor(),
            'mouth': MouthProcessor()
        }
        self.processed_points: dict = {}

    def main(self, points: dict):
        self.processed_points = {}
        for feature, processor in self.processors.items():
            feature_points = points.get(feature, {})
            self.processed_points[feature] = processor.process(feature_points)
        return self.processed_points

