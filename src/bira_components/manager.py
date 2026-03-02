from bira_components import SLM_Manager
from bira_components.camera import Camera
from bira_components.computer_vision import ComputerVision
from bira_components.mediator import BiraMediator
from bira_components.micro import Micro
from bira_components.speech_to_text import SpeechToText
from bira_components.states import IdleState
from bira_components.text_to_speech import TextToSpeech
from bira_components.uart_transmitter import UARTTransmitter


class BIRA_Manager:
    def __init__(self):
        # self.mediator = BiraMediator()
        self.camera = Camera(self.mediator)
        self.computer_vision = ComputerVision(self.mediator)
        self.uart_transmitter = UARTTransmitter(self.mediator)
        self.micro = Micro(self.mediator)
        self.text_to_speech = TextToSpeech(self.mediator)
        self.speech_to_text = SpeechToText(self.mediator)
        self.slm_manager = SLM_Manager(self.mediator)
        self.state = IdleState(self)
        self.context = {
            "objects_detected": [],
            "user_input": None,
            "feedback": None,
            "object_selected": None
        }
        self.counter = 0
    
    def change_state(self, new_state):
        self.state = new_state
        print(f"State changed to: {self.state.get_name()}")

    def handle(self, message):
        self.state.handle(message)
        pass