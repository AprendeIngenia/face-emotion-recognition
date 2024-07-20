from typing import Any, Tuple

import cv2
import mediapipe as mp
import numpy as np


class FaceMeshMediapipe:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.config_draw = self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

        self.face_mesh_object = mp.solutions.face_mesh
        self.face_mesh_mp = self.face_mesh_object.FaceMesh(static_image_mode=False,
                                                           max_num_faces=1,
                                                           refine_landmarks=True,
                                                           min_detection_confidence=0.6,
                                                           min_tracking_confidence=0.6)
        self.eye_brows_points: dict = {}
        self.eyes_points: dict = {}
        self.nose_points: dict = {}
        self.mouth_points: dict = {}

        self.mesh_points: list = []

    def face_mesh_inference(self, face_image: np.ndarray) -> Tuple[bool, Any]:
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        face_mesh = self.face_mesh_mp.process(rgb_image)
        if face_mesh.multi_face_landmarks is None:
            return False, face_mesh
        else:
            return True, face_mesh

    def extract_face_mesh_points(self, face_image: np.ndarray, face_mesh_info: Any) -> list:
        h, w, c = face_image.shape
        self.mesh_points = []
        for face_mesh in face_mesh_info.multi_face_landmarks:
            for i, points in enumerate(face_mesh.landmark):
                x, y = int(points.x * w), int(points.y * h)
                self.mesh_points.append([i, x, y])

        return self.mesh_points

    def draw_face_mesh(self, face_image: np.ndarray, face_mesh_info: Any, color: Tuple[int, int, int]):
        self.config_draw = self.mp_draw.DrawingSpec(color=color, thickness=1, circle_radius=1)
        for face_mesh in face_mesh_info.multi_face_landmarks:
            self.mp_draw.draw_landmarks(face_image, face_mesh, self.face_mesh_object.FACEMESH_TESSELATION,
                                        self.config_draw, self.config_draw)

    def extract_eye_brows_points(self, face_points: list) -> dict:
        self.eye_brows_points = {'right eyebrow arch': [], 'left eyebrow arch': [], 'distance eyebrow eye right': [],
                                 'distance eyebrow eye left': [], 'distance between eyebrows': []}
        if len(face_points) == 478:
            right_eyebrow_arch_indices = [46, 53, 52, 65, 55]
            left_eyebrow_arch_indices = [276, 283, 282, 295, 285]
            distance_eyebrow_eye_right_indices = [65, 468]
            distance_eyebrow_eye_left_indices = [295, 473]
            distance_between_eyebrows_indices = [55, 285]

            def get_eyebrow_points(indices):
                return [face_points[i][1:] for i in indices]

            right_eyebrow_arch_points = get_eyebrow_points(right_eyebrow_arch_indices)
            left_eyebrow_arch_points = get_eyebrow_points(left_eyebrow_arch_indices)
            distance_eyebrow_eye_right_points = get_eyebrow_points(distance_eyebrow_eye_right_indices)
            distance_eyebrow_eye_left_points = get_eyebrow_points(distance_eyebrow_eye_left_indices)
            distance_between_eyebrows_points = get_eyebrow_points(distance_between_eyebrows_indices)

            self.eye_brows_points['right eyebrow arch'] = [point for point in right_eyebrow_arch_points]
            self.eye_brows_points['left eyebrow arch'] = [point for point in left_eyebrow_arch_points]
            self.eye_brows_points['distance eyebrow eye right'] = [point for point in distance_eyebrow_eye_right_points]
            self.eye_brows_points['distance eyebrow eye left'] = [point for point in distance_eyebrow_eye_left_points]
            self.eye_brows_points['distance between eyebrows'] = [point for point in distance_between_eyebrows_points]

        else:
            raise Exception(f"face_points len: {len(face_points)} != 478")
        return self.eye_brows_points

    def extract_eyes_points(self, face_points: list) -> dict:
        self.eyes_points = {'right eye': [], 'left eye': []}
        if len(face_points) == 478:
            right_eye_indices = [130, 246, 161, 160, 159, 158, 157, 173, 155]
            left_eye_indices = [263, 466, 388, 387, 386, 385, 384, 398, 362]

            def get_eye_points(indices):
                return [face_points[i][1:] for i in indices]

            right_eye_points = get_eye_points(right_eye_indices)
            left_eye_points = get_eye_points(left_eye_indices)

            self.eyes_points['right eye'] = [coord for point in right_eye_points for coord in point]
            self.eyes_points['left eye'] = [coord for point in left_eye_points for coord in point]

        else:
            raise Exception(f"face_points len: {len(face_points)} != 278")

        return self.eyes_points

    def extract_nose_points(self, face_points: list):
        self.nose_points = {'right side nose': [], 'left side nose': []}
        if len(face_points) == 478:
            right_side_nose_indices = [37, 72, 38, 82]
            left_side_nose_indices = [267, 302, 268, 312]

            def get_nose_points(indices):
                return [face_points[i][1:] for i in indices]

            right_nose_points = get_nose_points(right_side_nose_indices)
            left_nose_points = get_nose_points(left_side_nose_indices)

            self.nose_points['right side nose'] = [coord for point in right_nose_points for coord in point]
            self.nose_points['left side nose'] = [coord for point in left_nose_points for coord in point]
        else:
            raise Exception(f"face points len: {len(face_points)} != 278")
        return self.nose_points

    def extract_mouth_points(self, face_points: list):
        self.mouth_points = {'upper lips contour': [], 'lower lips contour': [], 'mouth opening': []}
        if len(face_points) == 478:
            upper_mouth_contour_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 306]
            lower_mouth_contour_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 306]
            mouth_opening_indices = [13, 14]

            def get_mouth_points(indices):
                return [face_points[i][1:] for i in indices]

            upper_mouth_points = get_mouth_points(upper_mouth_contour_indices)
            lower_mouth_points = get_mouth_points(lower_mouth_contour_indices)
            mouth_opening_points = get_mouth_points(mouth_opening_indices)

            self.mouth_points['upper lips contour'] = [coord for point in upper_mouth_points for coord in point]
            self.mouth_points['lower lips contour'] = [coord for point in lower_mouth_points for coord in point]
            self.mouth_points['mouth opening'] = [coord for point in mouth_opening_points for coord in point]

        else:
            raise Exception(f"face_points len: {len(face_points)} != 278")
        return self.mouth_points

    def main_process(self, face_image: np.ndarray, draw: bool) -> Tuple[dict, dict, dict, dict, bool, np.ndarray]:
        original_image = face_image.copy()
        face_mesh_check, face_mesh_info = self.face_mesh_inference(face_image)
        if face_mesh_check is False:
            self.__init__()
            return (self.eye_brows_points, self.eyes_points, self.nose_points, self.mouth_points, False,
                    original_image)
        else:
            # extract face points
            mesh_points = self.extract_face_mesh_points(face_image, face_mesh_info)
            # create dicts with points
            self.eye_brows_points = self.extract_eye_brows_points(mesh_points)
            self.eyes_points = self.extract_eyes_points(mesh_points)
            self.nose_points = self.extract_nose_points(mesh_points)
            self.mouth_points = self.extract_mouth_points(mesh_points)
            if draw:
                self.draw_face_mesh(face_image, face_mesh_info, color=(255, 255, 0))
            return (self.eye_brows_points, self.eyes_points, self.nose_points, self.mouth_points, True,
                    original_image)
