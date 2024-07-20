import numpy as np
from emotion_processor.face_mesh.mediapipe import FaceMeshMediapipe


class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshMediapipe()

    def video_stream_processing(self, face_image: np.ndarray):
        eye_brows_points, eyes_points, nose_points, mouth_points, control_process, original_image = (
            self.face_mesh.main_process(face_image, draw=True))
        print(f'\ncejas: {eye_brows_points} \nojos: {eyes_points} \nnariz: {nose_points} \nboca: {mouth_points}')
