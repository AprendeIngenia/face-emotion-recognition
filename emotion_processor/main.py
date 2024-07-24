import numpy as np
from emotion_processor.face_mesh.face_mesh_processor import FaceMeshProcessor
from emotion_processor.data_processing.main import PointsProcessing


class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshProcessor()
        self.data_processing = PointsProcessing()

    def frame_processing(self, face_image: np.ndarray):
        face_points, control_process, original_image = self.face_mesh.process(face_image, draw=True)
        if control_process:
            processed_features = self.data_processing.main(face_points)
            self.print_results(processed_features)
        else:
            Exception(f"No face mesh")

    def print_results(self, processed_points: dict):
        for feature, results in processed_points.items():
            print(f"Results for {feature}: {results}")
