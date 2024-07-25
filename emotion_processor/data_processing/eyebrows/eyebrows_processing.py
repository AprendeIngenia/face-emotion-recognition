import numpy as np
from abc import ABC, abstractmethod


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, point1, point2):
        pass


class EuclideanDistanceCalculator(DistanceCalculator):
    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))


class EyebrowArchCalculator(ABC):
    @abstractmethod
    def calculate_eyebrow_arch(self, eyebrow_points):
        pass


class PolynomialEyebrowArchCalculator(EyebrowArchCalculator):
    def calculate_eyebrow_arch(self, eyebrow_points):
        x = [point[0] for point in eyebrow_points]
        y = [point[1] for point in eyebrow_points]
        z = np.polyfit(x, y, 2)
        return z[0]


class EyeBrowsPointsProcessing:
    def __init__(self, arch_calculator: EyebrowArchCalculator, distance_calculator: DistanceCalculator):
        self.arch_calculator = arch_calculator
        self.distance_calculator = distance_calculator
        self.eyebrows: dict = {}

    def calculate_distances(self, eyebrows_points: dict):
        right_eyebrow_to_eye_distance = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][0], eyebrows_points['distances'][1])
        left_eyebrow_to_eye_distance = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][2], eyebrows_points['distances'][3])
        right_eyebrow_to_forehead_distance = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][4], eyebrows_points['distances'][5])
        left_eyebrow_to_forehead_distance = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][6], eyebrows_points['distances'][7])
        distance_between_eyebrows = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][8], eyebrows_points['distances'][9])
        distance_between_eyebrow_forehead = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][10], eyebrows_points['distances'][11])

        return (right_eyebrow_to_eye_distance, left_eyebrow_to_eye_distance, right_eyebrow_to_forehead_distance,
                left_eyebrow_to_forehead_distance, distance_between_eyebrows, distance_between_eyebrow_forehead)

    def main(self, eyebrows_points: dict):
        # calculate eyebrow arch
        right_eyebrow_arch = self.arch_calculator.calculate_eyebrow_arch(eyebrows_points['right arch'])
        left_eyebrow_arch = self.arch_calculator.calculate_eyebrow_arch(eyebrows_points['left arch'])
        self.eyebrows['arch_right'] = right_eyebrow_arch
        self.eyebrows['arch_left'] = left_eyebrow_arch

        # calculate distance between eyebrow and its eye
        (right_eye_distance, left_eye_distance, right_forehead_distance, left_forehead_distance, eyebrows_distance,
         eyebrow_distance_forehead) = (self.calculate_distances(eyebrows_points))
        self.eyebrows['eye_right_distance'] = right_eye_distance
        self.eyebrows['eye_left_distance'] = left_eye_distance
        self.eyebrows['forehead_right_distance'] = right_forehead_distance
        self.eyebrows['forehead_left_distance'] = left_forehead_distance
        self.eyebrows['eyebrows_distance'] = eyebrows_distance
        self.eyebrows['eyebrow_distance_forehead'] = eyebrow_distance_forehead
        #print(f'Eyebrows: { {k: (round(float(v),4)) for k,v in self.eyebrows.items()}}')
        return self.eyebrows
