from emotion_processor.data_processing.feature_processor import FeatureProcessor
from emotion_processor.data_processing.eyebrows.eyebrows_processing import (EyeBrowsPointsProcessing,
                                                                            PolynomialEyebrowArchCalculator,
                                                                            EuclideanDistanceCalculator)


class EyeBrowsProcessor(FeatureProcessor):
    def __init__(self):
        arch_calculator = PolynomialEyebrowArchCalculator()
        distance_calculator = EuclideanDistanceCalculator()
        self.processor = EyeBrowsPointsProcessing(arch_calculator, distance_calculator)

    def process(self, points: dict):
        return self.processor.main(points)
