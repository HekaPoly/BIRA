from bira_components import SLM_Manager
from bira_components.camera import Camera
from bira_components.computer_vision import ComputerVision
from bira_components.mediator import BiraMediator
from bira_components.micro import Micro
from bira_components.speech_to_text import SpeechToText
from bira_components.states import IdleState
from bira_components.text_to_speech import TextToSpeech
from bira_components.uart_transmitter import UARTTransmitter

DEFAULT_CONTEXT = {
            "sl_object": [],
            "detection_labels": [],
            "user_input": "",
            "feedback": "",
            "response_code": 0,
            "object_selected": None
        }

class BIRA_Manager:
    def __init__(self):
        # self.mediator = BiraMediator()
        self.camera = Camera(self.mediator)
        self.computer_vision = ComputerVision(self.mediator)
        self.uart_transmitter = UARTTransmitter(self.mediator)
        self.micro = Micro()
        self.text_to_speech = TextToSpeech(self.mediator)
        self.speech_to_text = SpeechToText(self.mediator)
        self.slm_manager = SLM_Manager(self.mediator)

        self.state = IdleState(self)
        self._data = DEFAULT_CONTEXT.copy()
        self.counter = 0
    
    def change_state(self, new_state):
        self.state = new_state
        print(f"State changed to: {self.state.__class__.__name__}")
    
    def set_objects_detected(self, objects):
        self._data["objects_detected"] = objects
    
    def set_user_input(self, user_input: str):
        self._data["user_input"] = user_input
    
    def set_feedback(self, feedback):
        self._data["feedback"] = feedback
    
    def set_object_selected(self, obj):
        self._data["object_selected"] = obj
    
    def get_data(self):
        return self._data

    def reset_data(self):
        self._data = DEFAULT_CONTEXT.copy()

    def _handle(self):
        self.state.handle(self.data)

    def run(self):
        while True:
            self._handle()

