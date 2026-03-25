from .context import BIRA_Context
from .controller import BIRA_Controller
from bira_orchestration import states
from bira_orchestration.enums import StateCode

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
        self._context = BIRA_Context()
        self.counter = 0
    
    def add_user_input(self, user_input: str):
        self._context.user_inputs.append(user_input)

    def add_feedback(self, feedback):
        self._context.feedbacks.append(feedback)
    
    def reset_data(self):
        self._context.reset_all()

    def prepare_listening(self):
        self._context.reset_for_listening()

    def prepare_vision(self):
        self._context.reset_for_vision()

    def prepare_planning(self):
        self._context.reset_for_planning()

    def prepare_execution(self):
        self._context.reset_for_execution()
    
    def get_data(self):
        return self._context
    
    def get_last_user_input(self):
        if self._context.user_inputs:
            return self._context.user_inputs[-1]
        return None
    
    def get_last_feedback(self):
        if self._context.feedbacks:
            return self._context.feedbacks[-1]
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

