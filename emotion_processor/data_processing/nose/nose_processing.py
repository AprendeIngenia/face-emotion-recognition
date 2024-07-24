import numpy as np
from abc import ABC, abstractmethod


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, point1, point2):
        pass


class EuclideanDistanceCalculator(DistanceCalculator):
    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))


class NosePointsProcessing:
    def __init__(self, distance_calculator: DistanceCalculator):
        self.distance_calculator = distance_calculator
        self.nose: dict = {}

    def calculate_distances(self, eyebrows_points: dict):
        upper_mouth = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][0], eyebrows_points['distances'][1])
        lower_nose = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][2], eyebrows_points['distances'][3])

        return upper_mouth, lower_nose

    def main(self, mouth_points: dict):
        # calculate distance between nose and mouth
        mouth_upper_distance, nose_lower_distance = self.calculate_distances(mouth_points)
        self.nose['mouth_upper_distance'] = mouth_upper_distance
        self.nose['nose_lower_distance'] = nose_lower_distance
        #print(f'Nose: { {k: (round(float(v), 4)) for k, v in self.nose.items()} }')
        return self.nose
