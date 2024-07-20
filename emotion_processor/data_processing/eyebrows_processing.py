import numpy as np


class EyeBrowsPointsProcessing:
    def __init__(self):
        self.eyebrows: dict = {}
        self.right_eyebrow_arch = 0.0
        self.left_eyebrow_arch = 0.0
        self.right_eye_distance = 0.0
        self.left_eye_distance = 0.0
        self.eyebrows_distance = 0.0

    def calculate_eyebrow_arch(self, eyebrow_points):
        x = [point[0] for point in eyebrow_points]
        y = [point[1] for point in eyebrow_points]
        z = np.polyfit(x, y, 2)
        return z[0]

    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def calculate_distances(self, eyebrows_points: dict):
        right_eyebrow_to_eye_distance = self.calculate_distance(eyebrows_points['distance eyebrow eye right'][0],
                                                                eyebrows_points['distance eyebrow eye right'][1])
        left_eyebrow_to_eye_distance = self.calculate_distance(eyebrows_points['distance eyebrow eye left'][0],
                                                               eyebrows_points['distance eyebrow eye left'][1])
        distance_between_eyebrows = self.calculate_distance(eyebrows_points['distance between eyebrows'][0],
                                                            eyebrows_points['distance between eyebrows'][1])

        return right_eyebrow_to_eye_distance, left_eyebrow_to_eye_distance, distance_between_eyebrows

    def main(self, eyebrows_points: dict):
        # calculate eyebrow arch
        self.right_eyebrow_arch = self.calculate_eyebrow_arch(eyebrows_points['right eyebrow arch'])
        self.left_eyebrow_arch = self.calculate_eyebrow_arch(eyebrows_points['left eyebrow arch'])
        self.eyebrows['arch_right'] = self.right_eyebrow_arch
        self.eyebrows['left_right'] = self.left_eyebrow_arch
        # calculate distance between eyebrow and its eye
        self.right_eye_distance, self.left_eye_distance, self.eyebrows_distance = (
            self.calculate_distances(eyebrows_points))
        self.eyebrows['eye right distance'] = self.right_eye_distance
        self.eyebrows['eye left distance'] = self.left_eye_distance
        self.eyebrows['eyebrows distance'] = self.eyebrows_distance
        return self.eyebrows
