from emotion_processor.data_processing.feature_processor import FeatureProcessor
from emotion_processor.data_processing.eyebrows.eyebrows_processor import EyeBrowsProcessor
from emotion_processor.data_processing.eyes.eyes_processor import EyesProcessor
from emotion_processor.data_processing.nose.nose_processor import NoseProcessor
from emotion_processor.data_processing.mouth.mouth_processor import MouthProcessor


class PointsProcessing:
    def __init__(self):
        self.processors: dict[str, FeatureProcessor] = {
            'eye_brows': EyeBrowsProcessor(),
            'eyes': EyesProcessor(),
            'nose': NoseProcessor(),
            'mouth': MouthProcessor()
        }

    def main(self, points: dict):
        for feature, processor in self.processors.items():
            feature_points = points.get(feature, {})
            processor.process(feature_points)

