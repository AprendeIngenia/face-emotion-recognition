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
        self.eye_brows_points = {'right_eyebrow': [], 'left_eyebrow': []}
        if len(face_points) == 478:
            # right eyebrow
            eb_r1x, eb_r1y = face_points[46][1:]
            eb_r2x, eb_r2y = face_points[53][1:]
            eb_r3x, eb_r3y = face_points[52][1:]
            eb_r4x, eb_r4y = face_points[65][1:]
            eb_r5x, eb_r5y = face_points[55][1:]
            # left eyebrow
            eb_l1x, eb_l1y = face_points[276][1:]
            eb_l2x, eb_l2y = face_points[283][1:]
            eb_l3x, eb_l3y = face_points[282][1:]
            eb_l4x, eb_l4y = face_points[295][1:]
            eb_l5x, eb_l5y = face_points[285][1:]

            self.eye_brows_points['right_eyebrow'].append([eb_r1x, eb_r1y, eb_r2x, eb_r2y, eb_r3x, eb_r3y, eb_r4x,
                                                           eb_r4y, eb_r5x, eb_r5y])
            self.eye_brows_points['left_eyebrow'].append([eb_l1x, eb_l1y, eb_l2x, eb_l2y, eb_l3x, eb_l3y, eb_l4x,
                                                          eb_l4y, eb_l5x, eb_l5y])
            return self.eye_brows_points

    def extract_eyes_points(self, face_points: list) -> dict:
        self.eyes_points = {'right_eye': [], 'left_eye': []}
        if len(face_points) == 478:
            # right eye
            # upper right eyelid
            e_r1x, e_r1y = face_points[130][1:]
            e_r2x, e_r2y = face_points[246][1:]
            e_r3x, e_r3y = face_points[161][1:]
            e_r4x, e_r4y = face_points[160][1:]
            e_r5x, e_r5y = face_points[159][1:]
            e_r6x, e_r6y = face_points[158][1:]
            e_r7x, e_r7y = face_points[157][1:]
            e_r8x, e_r8y = face_points[173][1:]
            e_r9x, e_r9y = face_points[155][1:]

            # left eye
            # upper left eyelid
            e_l1x, e_l1y = face_points[263][1:]
            e_l2x, e_l2y = face_points[466][1:]
            e_l3x, e_l3y = face_points[388][1:]
            e_l4x, e_l4y = face_points[387][1:]
            e_l5x, e_l5y = face_points[386][1:]
            e_l6x, e_l6y = face_points[385][1:]
            e_l7x, e_l7y = face_points[384][1:]
            e_l8x, e_l8y = face_points[398][1:]
            e_l9x, e_l9y = face_points[362][1:]

            self.eyes_points['right_eye'].append([e_r1x, e_r1y, e_r2x, e_r2y, e_r3x, e_r3y, e_r4x, e_r4y, e_r5x, e_r5y,
                                                  e_r6x, e_r6y, e_r7x, e_r7y, e_r8x, e_r8y, e_r9x, e_r9y])
            self.eyes_points['left_eye'].append([e_l1x, e_l1y, e_l2x, e_l2y, e_l3x, e_l3y, e_l4x, e_l4y, e_l5x, e_l5y,
                                                 e_l6x, e_l6y, e_l7x, e_l7y, e_l8x, e_l8y, e_l9x, e_l9y])
            return self.eyes_points

    def extract_nose_points(self, face_points: list):
        self.nose_points = {'right_side_nose': [], 'left_side_nose': []}
        if len(face_points) == 478:
            # right side nose
            n_r1x, n_r1y = face_points[37][1:]
            n_r2x, n_r2y = face_points[72][1:]
            n_r3x, n_r3y = face_points[38][1:]
            n_r4x, n_r4y = face_points[82][1:]

            # left side nose
            n_l1x, n_l1y = face_points[267][1:]
            n_l2x, n_l2y = face_points[302][1:]
            n_l3x, n_l3y = face_points[268][1:]
            n_l4x, n_l4y = face_points[312][1:]

            self.nose_points['right_side_nose'].append([n_r1x, n_r1y, n_r2x, n_r2y, n_r3x, n_r3y, n_r4x, n_r4y])
            self.nose_points['left_side_nose'].append([n_l1x, n_l1y, n_l2x, n_l2y, n_l3x, n_l3y, n_l4x, n_l4y])
            return self.nose_points

    def extract_mouth_points(self, face_points: list):
        self.mouth_points = {'upper mouth contour': [], 'lower mouth contour': [], 'mouth opening': []}
        if len(face_points) == 478:
            # mouth
            # upper mouth contour
            m_uc1x, m_uc1y = face_points[61][1:]
            m_uc2x, m_uc2y = face_points[185][1:]
            m_uc3x, m_uc3y = face_points[40][1:]
            m_uc4x, m_uc4y = face_points[39][1:]
            m_uc5x, m_uc5y = face_points[37][1:]
            m_uc6x, m_uc6y = face_points[0][1:]
            m_uc7x, m_uc7y = face_points[267][1:]
            m_uc8x, m_uc8y = face_points[269][1:]
            m_uc9x, m_uc9y = face_points[270][1:]
            m_uc10x, m_uc10y = face_points[409][1:]
            m_uc11x, m_uc11y = face_points[306][1:]
            # lower mouth contour
            m_lc1x, m_lc1y = face_points[61][1:]
            m_lc2x, m_lc2y = face_points[146][1:]
            m_lc3x, m_lc3y = face_points[91][1:]
            m_lc4x, m_lc4y = face_points[181][1:]
            m_lc5x, m_lc5y = face_points[84][1:]
            m_lc6x, m_lc6y = face_points[17][1:]
            m_lc7x, m_lc7y = face_points[314][1:]
            m_lc8x, m_lc8y = face_points[405][1:]
            m_lc9x, m_lc9y = face_points[321][1:]
            m_lc10x, m_lc10y = face_points[375][1:]
            m_lc11x, m_lc11y = face_points[306][1:]
            # mouth opening
            m_o1x, m_o1y = face_points[13][1:]
            m_o2x, m_o2y = face_points[14][1:]

            self.mouth_points['upper mouth contour'].append([m_uc1x, m_uc1y, m_uc2x, m_uc2y, m_uc3x, m_uc3y, m_uc4x,
                                                             m_uc4y, m_uc5x, m_uc5y, m_uc6x, m_uc6y, m_uc7x, m_uc7y,
                                                             m_uc8x, m_uc8y, m_uc9x, m_uc9y, m_uc10x, m_uc10y, m_uc11x,
                                                             m_uc11y])
            self.mouth_points['lower mouth contour'].append([m_lc1x, m_lc1y, m_lc2x, m_lc2y, m_lc3x, m_lc3y, m_lc4x,
                                                             m_lc4y, m_lc5x, m_lc5y, m_lc6x, m_lc6y, m_lc7x, m_lc7y,
                                                             m_lc8x, m_lc8y, m_lc9x, m_lc9y, m_lc10x, m_lc10y, m_lc11x,
                                                             m_lc11y])
            self.mouth_points['mouth opening'].append([m_o1x, m_o1y, m_o2x, m_o2y])
            return self.mouth_points

    def main_process(self, face_image: np.ndarray, draw: bool) -> Tuple[dict, dict, dict, dict, str, np.ndarray]:
        original_image = face_image.copy()
        face_mesh_check, face_mesh_info = self.face_mesh_inference(face_image)
        if face_mesh_check is False:
            return (self.eye_brows_points, self.eyes_points, self.nose_points, self.mouth_points, 'no face mesh',
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
            return (self.eye_brows_points, self.eyes_points, self.nose_points, self.mouth_points, 'face mesh',
                    original_image)
