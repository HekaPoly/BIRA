from bira_orchestration.context import BiraContext
from bira_orchestration.states.executing_state import ExecutingState
from bira_orchestration.states.exiting_state import ExitState
from bira_orchestration.states.idle_state import IdleState
from bira_orchestration.states.listening_state import ListeningState
from bira_orchestration.states.planning_state import PlanningState
from bira_orchestration.states.vision_state import VisionState
from bira_orchestration.enums import StateCode

STATE_CLASSES = {
    StateCode.IDLE: IdleState,
    StateCode.LISTENING: ListeningState,
    StateCode.VISION: VisionState,
    StateCode.PLANNING: PlanningState,
    StateCode.EXECUTING: ExecutingState,
    StateCode.EXIT: ExitState,
}

class BiraManager:
    def __init__(self, mock_mode: bool = False):
        if mock_mode:
            from bira_orchestration.mock_controller import MockedBiraController

            self.controller = MockedBiraController()
        else:
            from bira_orchestration.controller import BiraController

            self.controller = BiraController()
        self.state = IdleState(self)
        self._context = BiraContext()
        self.counter = 0

    def preload(self):
        self.controller.preload_components()
    
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
            print(f"[State log] Warning: Transitioning to a previous state ({next_code} < {self.state.code})")
            self.counter += 1
            if self.counter > 3:
                print("[State log] Looped three times. Forced exit.")
                next_code = StateCode.EXIT
        state_cls = STATE_CLASSES[next_code]
        self.state = state_cls(self)
        print(f"\n[State log] Entering {self.state} (Code {self.state.code})")

    def run(self):
        while True:
            self.state.handle()

