from abc import ABC, abstractmethod


class FeatureProcessor(ABC):
    @abstractmethod
    def process(self, points: dict):
        raise NotImplementedError
