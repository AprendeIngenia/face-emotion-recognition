from emotion_processor.data_processing.feature_processor import FeatureProcessor
from emotion_processor.data_processing.eyes.eyes_processing import (EyesPointsProcessing,
                                                                    PolynomialEyesArchCalculator,
                                                                    EuclideanDistanceCalculator)


class EyesProcessor(FeatureProcessor):
    def __init__(self):
        arch_calculator = PolynomialEyesArchCalculator()
        distance_calculator = EuclideanDistanceCalculator()
        self.processor = EyesPointsProcessing(arch_calculator, distance_calculator)

    def process(self, points: dict):
        return self.processor.main(points)
