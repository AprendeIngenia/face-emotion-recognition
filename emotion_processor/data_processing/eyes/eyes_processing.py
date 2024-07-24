import numpy as np
from abc import ABC, abstractmethod


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, point1, point2):
        pass


class EuclideanDistanceCalculator(DistanceCalculator):
    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))


class EyesArchCalculator(ABC):
    @abstractmethod
    def calculate_eyes_arch(self, eyebrow_points):
        pass


class PolynomialEyesArchCalculator(EyesArchCalculator):
    def calculate_eyes_arch(self, eyebrow_points):
        x = [point[0] for point in eyebrow_points]
        y = [point[1] for point in eyebrow_points]
        z = np.polyfit(x, y, 2)
        return z[0]


class EyesPointsProcessing:
    def __init__(self, arch_calculator: EyesArchCalculator, distance_calculator: DistanceCalculator):
        self.arch_calculator = arch_calculator
        self.distance_calculator = distance_calculator
        self.eyes: dict = {}

    def calculate_distances(self, eyebrows_points: dict):
        right_upper_eyelid = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][0], eyebrows_points['distances'][1])
        left_upper_eyelid = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][2], eyebrows_points['distances'][3])
        right_lower_eyelid = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][4], eyebrows_points['distances'][5])
        left_lower_eyelid = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][6], eyebrows_points['distances'][7])

        return right_upper_eyelid, left_upper_eyelid, right_lower_eyelid, left_lower_eyelid

    def main(self, eyes_points: dict):
        # calculate eyebrow arch
        right_eyes_arch = self.arch_calculator.calculate_eyes_arch(eyes_points['right arch'])
        left_eyes_arch = self.arch_calculator.calculate_eyes_arch(eyes_points['left arch'])
        self.eyes['arch_right'] = right_eyes_arch
        self.eyes['arch_left'] = left_eyes_arch

        # calculate distance between eyes and its eyelids
        (right_upper_eyelid_distance, left_upper_eyelid_distance, right_lower_eyelid_distance,
         left_lower_eyelid_distance) = (self.calculate_distances(eyes_points))
        self.eyes['right_upper_eyelid_distance'] = right_upper_eyelid_distance
        self.eyes['left_upper_eyelid_distance'] = left_upper_eyelid_distance
        self.eyes['right_lower_eyelid_distance'] = right_lower_eyelid_distance
        self.eyes['left_lower_eyelid_distance'] = left_lower_eyelid_distance
        #print(f'Eyes: { {k: (round(float(v),4)) for k,v in self.eyes.items()}}')
        return self.eyes
