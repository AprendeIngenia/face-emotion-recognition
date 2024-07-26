import numpy as np
from abc import ABC, abstractmethod


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, point1, point2):
        pass


class EuclideanDistanceCalculator(DistanceCalculator):
    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))


class MouthArchCalculator(ABC):
    @abstractmethod
    def calculate_lips_arch(self, eyebrow_points):
        pass


class PolynomialMouthArchCalculator(MouthArchCalculator):
    def calculate_lips_arch(self, eyebrow_points):
        x = [point[0] for point in eyebrow_points]
        y = [point[1] for point in eyebrow_points]
        z = np.polyfit(x, y, 2)
        return z[0]


class MouthPointsProcessing:
    def __init__(self, arch_calculator: MouthArchCalculator, distance_calculator: DistanceCalculator):
        self.arch_calculator = arch_calculator
        self.distance_calculator = distance_calculator
        self.mouth: dict = {}

    def calculate_distances(self, eyebrows_points: dict):
        upper_mouth = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][0], eyebrows_points['distances'][1])
        lower_mouth = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][2], eyebrows_points['distances'][3])
        right_smile = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][4], eyebrows_points['distances'][5])
        right_lip = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][6], eyebrows_points['distances'][7])
        left_smile = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][8], eyebrows_points['distances'][9])
        left_lip = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][10], eyebrows_points['distances'][11])

        return upper_mouth, lower_mouth, right_smile, right_lip, left_smile, left_lip

    def main(self, mouth_points: dict):
        # calculate eyebrow arch
        upper_arch = self.arch_calculator.calculate_lips_arch(mouth_points['upper arch'])
        lower_arch = self.arch_calculator.calculate_lips_arch(mouth_points['lower arch'])
        self.mouth['upper_arch'] = upper_arch
        self.mouth['lower_arch'] = lower_arch

        # calculate distance between lips
        (mouth_upper_distance, mouth_lower_distance, right_smile_distance, right_lip_distance, left_smile_distance,
         left_lip_distance) = self.calculate_distances(mouth_points)
        self.mouth['mouth_upper_distance'] = mouth_upper_distance
        self.mouth['mouth_lower_distance'] = mouth_lower_distance
        self.mouth['right_smile_distance'] = right_smile_distance
        self.mouth['right_lip_distance'] = right_lip_distance
        self.mouth['left_smile_distance'] = left_smile_distance
        self.mouth['left_lip_distance'] = left_lip_distance
        #print(f'Mouth: { {k: (round(float(v), 4)) for k, v in self.mouth.items()} }')
        return self.mouth
