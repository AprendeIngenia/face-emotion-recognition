import numpy as np
import cv2
import mediapipe as mp
from typing import Any, Tuple, List, Dict


class FaceMeshInference:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, image: np.ndarray) -> Tuple[bool, Any]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_mesh = self.face_mesh.process(rgb_image)
        return bool(face_mesh.multi_face_landmarks), face_mesh


class FaceMeshExtractor:
    def __init__(self):
        self.points: dict = {
            'eye_brows': {'right arch': [], 'left arch': [], 'distances': []},
            'eyes': {'right': [], 'left': []},
            'nose': {'right': [], 'left': []},
            'mouth': {'upper': [], 'lower': [], 'opening': []}
        }

    def extract_points(self, face_image: np.ndarray, face_mesh_info: Any) -> List[List[int]]:
        h, w, _ = face_image.shape
        mesh_points = [
            [i, int(pt.x * w), int(pt.y * h)]
            for face in face_mesh_info.multi_face_landmarks
            for i, pt in enumerate(face.landmark)
        ]
        return mesh_points

    def extract_feature_points(self, face_points: List[List[int]], feature_indices: dict):
        for feature, indices in feature_indices.items():
            for sub_feature, sub_indices in indices.items():
                self.points[feature][sub_feature] = [face_points[i][1:] for i in sub_indices]

    def get_eye_brows_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'eye_brows': {
                'right arch': [46, 53, 52, 65, 55],
                'left arch': [276, 283, 282, 295, 285],
                'distances': [65, 468, 295, 473, 69, 66, 299, 296, 55, 285]
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['eye_brows']

    def get_eyes_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'eyes': {
                'right': [130, 246, 161, 160, 159, 158, 157, 173, 155],
                'left': [263, 466, 388, 387, 386, 385, 384, 398, 362]
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['eyes']

    def get_nose_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'nose': {
                'right': [37, 72, 38, 82],
                'left': [267, 302, 268, 312]
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['nose']

    def get_mouth_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'mouth': {
                'upper': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 306],
                'lower': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 306],
                'opening': [13, 14]
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['mouth']


class FaceMeshDrawer:
    def __init__(self, color: Tuple[int, int, int] = (255, 255, 0)):
        self.mp_draw = mp.solutions.drawing_utils
        self.config_draw = self.mp_draw.DrawingSpec(color=color, thickness=1, circle_radius=1)

    def draw(self, face_image: np.ndarray, face_mesh_info: Any):
        for face_mesh in face_mesh_info.multi_face_landmarks:
            self.mp_draw.draw_landmarks(face_image, face_mesh, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                        self.config_draw, self.config_draw)


class FaceMeshProcessor:
    def __init__(self):
        self.inference = FaceMeshInference()
        self.extractor = FaceMeshExtractor()
        self.drawer = FaceMeshDrawer()

    def process(self, face_image: np.ndarray, draw: bool = True) -> Tuple[dict, bool, np.ndarray]:
        original_image = face_image.copy()
        success, face_mesh_info = self.inference.process(face_image)
        if not success:
            return {}, False, original_image

        face_points = self.extractor.extract_points(face_image, face_mesh_info)
        points = {
            'eye_brows': self.extractor.get_eye_brows_points(face_points),
            'eyes': self.extractor.get_eyes_points(face_points),
            'nose': self.extractor.get_nose_points(face_points),
            'mouth': self.extractor.get_mouth_points(face_points)
        }

        if draw:
            self.drawer.draw(face_image, face_mesh_info)

        return points, True, original_image
