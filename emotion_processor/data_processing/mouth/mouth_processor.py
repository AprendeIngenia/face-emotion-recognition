from emotion_processor.data_processing.feature_processor import FeatureProcessor
from emotion_processor.data_processing.mouth.mouth_processing import (MouthPointsProcessing,
                                                                      PolynomialMouthArchCalculator,
                                                                      EuclideanDistanceCalculator)


class MouthProcessor(FeatureProcessor):
    def __init__(self):
        arch_calculator = PolynomialMouthArchCalculator()
        distance_calculator = EuclideanDistanceCalculator()
        self.processor = MouthPointsProcessing(arch_calculator, distance_calculator)

    def process(self, points: dict):
        return self.processor.main(points)
