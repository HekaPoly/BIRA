# from bira_components import SLM_Manager
# from bira_components.camera import Camera
# from bira_components.computer_vision import ComputerVision
# from bira_components.mediator import BiraMediator
# from bira_components.micro import Micro
# from bira_components.speech_to_text import SpeechToText
import bira_components.states as states
# from bira_components.text_to_speech import TextToSpeech
# from bira_components.uart_transmitter import UARTTransmitter

DEFAULT_CONTEXT = {
            "sl_object": [],
            "detection_labels": [],
            "user_input": "",
            "feedback": "",
            "response_code": 0,
            "object_selected": None,
            "user_wants_idle_state": False
        }

class BIRA_Manager:
    def __init__(self):
        # self.mediator = BiraMediator()
        # self.camera = Camera(self.mediator)
        # self.computer_vision = ComputerVision(self.mediator)
        # self.uart_transmitter = UARTTransmitter(self.mediator)
        # self.micro = Micro()
        # self.text_to_speech = TextToSpeech(self.mediator)
        # self.speech_to_text = SpeechToText(self.mediator)
        # self.slm_manager = SLM_Manager(self.mediator)

        self.state = states.IdleState(self)
        self._data = DEFAULT_CONTEXT.copy()
        self.counter = 0
    
    def change_state(self, new_state):
        self.state = new_state
        print(f"State changed to: {self.state}")
    
    def set_objects_detected(self, objects):
        self._data["objects_detected"] = objects
    
    def set_user_input(self, user_input: str):
        self._data["user_input"] = user_input
    
    def set_feedback(self, feedback):
        self._data["feedback"] = feedback
    
    def set_object_selected(self, obj):
        self._data["object_selected"] = obj
    
    def set_response_code(self, code):
        self._data["response_code"] = code

    def set_user_wants_idle_state(self, wants_idle_state: bool):
        self._data["user_wants_idle_state"] = wants_idle_state

    def get_data(self):
        return self._data

    def reset_data(self):
        self._data = DEFAULT_CONTEXT.copy()

    def _handle(self):
        self.state.handle()

    def run(self):
        while True:
            self._handle()

