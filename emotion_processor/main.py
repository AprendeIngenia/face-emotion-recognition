import numpy as np
from emotion_processor.face_mesh.face_mesh_processor import FaceMeshProcessor
from emotion_processor.data_processing.main import PointsProcessing


class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshProcessor()
        self.data_processing = PointsProcessing()

    def video_stream_processing(self, face_image: np.ndarray):
        face_points, control_process, original_image = (
            self.face_mesh.process(face_image, draw=True))
        if control_process:
            self.data_processing.main(face_points)
        else:
            Exception(f"No face mesh")
