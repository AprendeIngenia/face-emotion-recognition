import numpy as np
from emotion_processor.face_mesh.mediapipe import FaceMeshMediapipe
from emotion_processor.data_processing.main import PointsProcessing


class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshMediapipe()
        self.data_processing = PointsProcessing()

    def video_stream_processing(self, face_image: np.ndarray):
        eye_brows_points, eyes_points, nose_points, mouth_points, control_process, original_image = (
            self.face_mesh.main_process(face_image, draw=True))
        if control_process:
            self.data_processing.main(eye_brows_points, eyes_points, nose_points, mouth_points)
        else:
            Exception(f"No face mesh")
