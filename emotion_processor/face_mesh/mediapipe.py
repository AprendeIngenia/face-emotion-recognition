import numpy as np
import cv2
import mediapipe as mp
from typing import Any, Tuple


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

    def extract_face_mesh_points(self, face_image: np.ndarray, face_mesh_info: Any, viz: bool) -> list:
        h, w, c = face_image.shape
        self.mesh_points = []
        for face_mesh in face_mesh_info.multi_face_landmarks:
            for i, points in enumerate(face_mesh.landmark):
                x, y = int(points.x * w), int(points.y * h)
                self.mesh_points.append([i, x, y])

            if viz:
                self.mp_draw.draw_landmarks(face_image, face_mesh, self.face_mesh_object.FACEMESH_TESSELATION,
                                            self.config_draw, self.config_draw)

        return self.mesh_points

    def extract_eye_brows_points(self, face_points: list, face_image: np.ndarray) -> dict:
        print(len(face_points))
        if len(face_points) == 478:
            eb_rfx, eb_rfy = face_points[46][1:]
            cv2.circle(face_image, (eb_rfx, eb_rfy), 5, (0, 0, 0), -1)

    def main_process(self, face_image: np.ndarray) -> Tuple[dict, dict, dict, dict, str, np.ndarray]:
        original_image = face_image.copy()
        face_mesh_check, face_mesh_info = self.face_mesh_inference(face_image)
        if face_mesh_check is False:
            return (self.eye_brows_points, self.eyes_points, self.nose_points, self.mouth_points, 'no face mesh',
                    original_image)
        else:
            mesh_points = self.extract_face_mesh_points(face_image, face_mesh_info, viz=True)
            self.extract_eye_brows_points(mesh_points, face_image)
            return (self.eye_brows_points, self.eyes_points, self.nose_points, self.mouth_points, 'face mesh',
                    original_image)



