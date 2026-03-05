# from bira_components import SLM_Manager
# from bira_components.camera import Camera
# from bira_components.computer_vision import ComputerVision
# from bira_components.mediator import BiraMediator
# from bira_components.micro import Micro
# from bira_components.speech_to_text import SpeechToText
import bira_components.states as states
from bira_components.codes import ListeningCode, VisionCode, PlanificationCode, ExecutionCode
# from bira_components.text_to_speech import TextToSpeech
# from bira_components.uart_transmitter import UARTTransmitter

DEFAULT_CONTEXT = {
            "objects_detected": [],
            "detection_labels": [],
            "user_input": "",
            "feedback": "",
            "object_selected": None,
            "listening_code": ListeningCode.NO_RESPONSE,
            "vision_code": VisionCode.NO_RESPONSE,
            "planification_code": PlanificationCode.NO_RESPONSE,
            "execution_code": ExecutionCode.NO_RESPONSE,
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
    
    def increment_counter(self):
        self.counter += 1
        print(f"Counter incremented to {self.counter}")
        if self.counter > 3:
            print("Looped three times. Forced exit.")
            self.change_state(states.ExitState(self))
            self.state.handle()  # Handle the exit state immediately
    
    def set_objects_detected(self, objects, labels):
        self._data["objects_detected"] = objects
        self._data["detection_labels"] = labels

    def set_user_input(self, user_input: str):
        self._data["user_input"] = user_input
    
    def set_feedback(self, feedback):
        self._data["feedback"] = feedback
    
    def set_object_selected(self, obj):
        self._data["object_selected"] = obj
    
    def set_listening_code(self, code: ListeningCode):
        self._data["listening_code"] = code
    
    def set_vision_code(self, code: VisionCode):
        self._data["vision_code"] = code
    
    def set_planification_code(self, code: PlanificationCode):
        self._data["planification_code"] = code
    
    def set_execution_code(self, code: ExecutionCode):
        self._data["execution_code"] = code

    def get_data(self):
        return self._data

    def reset_data(self):
        self._data = DEFAULT_CONTEXT.copy()
    
    def change_state(self, new_state):
        self.state = new_state
        print(f"Entering {self.state}")

    def run(self):
        while True:
            self.state.handle()

