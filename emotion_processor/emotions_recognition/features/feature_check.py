from abc import ABC, abstractmethod


class EyebrowsCheck(ABC):
    @abstractmethod
    def check_eyebrows(self, eyebrows: dict) -> str:
        pass


class EyesCheck(ABC):
    @abstractmethod
    def check_eyes(self, eyes: dict) -> str:
        pass


class NoseCheck(ABC):
    @abstractmethod
    def check_nose(self, nose: dict) -> str:
        pass


class MouthCheck(ABC):
    @abstractmethod
    def check_mouth(self, mouth: dict) -> str:
        pass
