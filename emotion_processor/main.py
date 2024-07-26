import numpy as np
from emotion_processor.face_mesh.face_mesh_processor import FaceMeshProcessor
from emotion_processor.data_processing.main import PointsProcessing
from emotion_processor.emotions_recognition.main import EmotionRecognition


class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshProcessor()
        self.data_processing = PointsProcessing()
        self.emotions_recognition = EmotionRecognition()

    def frame_processing(self, face_image: np.ndarray):
        face_points, control_process, original_image = self.face_mesh.process(face_image, draw=True)
        if control_process:
            processed_features = self.data_processing.main(face_points)
            emotion = self.emotions_recognition.recognize_emotion(processed_features)
        else:
            Exception(f"No face mesh")
