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
            'eye_brows': {'right arch': [], 'left arch': [], 'distances': {}},
            'eyes': {'right arch': [], 'left arch': []},
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
            if isinstance(indices, dict):
                for sub_feature, sub_indices in indices.items():
                    if isinstance(sub_indices, dict):
                        for sub_sub_feature, sub_sub_indices in sub_indices.items():
                            self.points[feature][sub_feature][sub_sub_feature] = [face_points[i][1:] for i in sub_sub_indices]
                    else:
                        self.points[feature][sub_feature] = [face_points[i][1:] for i in sub_indices]
            else:
                self.points[feature] = [face_points[i][1:] for i in indices]

    def get_eye_brows_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'eye_brows': {
                'right arch': [46, 53, 52, 65, 55],
                'left arch': [276, 283, 282, 295, 285],
                'distances': {
                    "right_eyebrow_to_eye": [65, 468],
                    "left_eyebrow_to_eye": [295, 473],
                    "right_eyebrow_to_forehead": [299, 296],
                    "between_eyebrows": [296, 55],
                    "left_eyebrow_to_forehead": [55, 285],
                }
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['eye_brows']

    def get_eyes_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'eyes': {
                'right arch': [33, 246, 161, 160, 159, 158, 157, 173, 133],
                'left arch': [263, 398, 384, 385, 386, 387, 388, 466, 263],
                'distances': [159, 145, 385, 374, 145, 230, 374, 450],
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['eyes']

    def get_nose_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'nose': {
                'distances': [0, 13, 2, 164],
            }
        }
        self.extract_feature_points(face_points, feature_indices)
        return self.points['nose']

    def get_mouth_points(self, face_points: List[List[int]]) -> Dict[str, List[List[int]]]:
        feature_indices = {
            'mouth': {
                'upper arch': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
                'lower arch': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
                'distances': [13, 14, 17, 200]
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
