from __future__ import annotations
from abc import ABC

from camera import Camera
from computer_vision import ComputerVision
from uart_transmitter import UARTTransmitter
from micro import Micro
from text_to_speech import TextToSpeech
from speech_to_text import SpeechToText
    

class Mediator(ABC):
    """
    The Mediator interface declares a method used by components to notify the
    mediator about various events. The Mediator may react to these events and
    pass the execution to other components.
    """

    def notify(self, sender: object, event: str) -> None:
        pass

class BIRAManager(Mediator):
    def __init__(self):
        self.camera = Camera()
        self.computer_vision = ComputerVision()
        self.uart_transmitter = UARTTransmitter()
        self.micro = Micro()
        self.text_to_speech = TextToSpeech()
        self.speech_to_text = SpeechToText()
        self.slm = None

    def notify(self, sender: object, event: str) -> None:
        if event == "wait_for_volume":
            print("Waiting for volume..")
            self.micro.wait_for_volume()
            
        elif event == "volume_detected":
            print("Volume detected, starting speech recognition...")
            
            command = self.speech_to_text.recognize_speech()
            
            print(f"Recognized command: {command}")
            self.mediator.notify(self, "command_recognized")
            
            
            
        if event == "wake_up":
            pass
        
        
class BiraComponent:
    """
    It is the base component that provides the basic functionality of storing a mediator's
    instance inside component objects.
    """

    def __init__(self, mediator: Mediator = None) -> None:
        self._mediator = mediator

    @property
    def mediator(self) -> Mediator:
        return self._mediator

    @mediator.setter
    def mediator(self, mediator: Mediator) -> None:
        self._mediator = mediator