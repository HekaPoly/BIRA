from bira_components.controller import BIRA_Controller
import bira_components.states as states
from bira_components.enums import (
    ListeningCode,
    VisionCode,
    PlanificationCode,
    ExecutionCode,
    StateCode,
)

# Context is managed by the BIRA_CONTROLLER, response codes are managed by states.
DEFAULT_CONTEXT = {
            "objects_detected": [],
            "detection_labels": [],
            "user_inputs": [],
            "feedbacks": [],
            "object_selected": None,
            "listening_code": ListeningCode.NO_RESPONSE,
            "vision_code": VisionCode.NO_RESPONSE,
            "planification_code": PlanificationCode.NO_RESPONSE,
            "execution_code": ExecutionCode.NO_RESPONSE,
        }

STATE_CLASSES = {
    StateCode.IDLE: states.IdleState,
    StateCode.LISTENING: states.ListeningState,
    StateCode.VISION: states.VisionState,
    StateCode.PLANNING: states.PlanningState,
    StateCode.EXECUTING: states.ExecutingState,
    StateCode.EXIT: states.ExitState,
}

class BIRA_Manager:
    def __init__(self):
        self.controller = BIRA_Controller()
        self.state = states.IdleState(self)
        self._data = DEFAULT_CONTEXT.copy()
        self.counter = 0
    
    def set_objects_detected(self, objects, labels):
        self._data["objects_detected"] = objects
        self._data["detection_labels"] = labels

    def add_user_input(self, user_input: str):
        self._data["user_inputs"].append(user_input)

    def add_feedback(self, feedback):
        self._data["feedbacks"].append(feedback)
    
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

    def reset_data(self):
        self._data = DEFAULT_CONTEXT.copy()
    
    def get_data(self):
        return self._data
    
    def get_last_user_input(self):
        if self._data["user_inputs"]:
            return self._data["user_inputs"][-1]
        return None
    
    def get_last_feedback(self):
        if self._data["feedbacks"]:
            return self._data["feedbacks"][-1]
        return None

    def change_state(self, next_code: StateCode):
        if next_code < self.state.code:
            print(f"Warning: Transitioning to a previous state ({next_code} < {self.state.code})")
            self.counter += 1
            if self.counter > 3:
                print("Looped three times. Forced exit.")
                next_code = StateCode.EXIT
        state_cls = STATE_CLASSES[next_code]
        self.state = state_cls(self)
        print(f"\nEntering {self.state} (Code {self.state.code})")

    def run(self):
        while True:
            self.state.handle()

