from bira_orchestration.states.base_state import State
from bira_orchestration.enums import StateCode

class IdleState(State):
    code = StateCode.IDLE

    def __str__(self):
        return "IdleState"

    def _prepare(self):
        self.bira_manager.reset_data()

    def _handle(self):
        self.bira_manager.controller.sleep_mode()

    def _decide_next_state(self):
        self.log_state("Wake up")
        feedback = "I am awake. What can I do for you?"
        self.emit_feedback(feedback, source="state_log")
        self.bira_manager.change_state(StateCode.LISTENING)