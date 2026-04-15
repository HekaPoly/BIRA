from bira_orchestration.states.base_state import State
from bira_orchestration.enums import StateCode, ListeningCode

class ListeningState(State):
    code = StateCode.LISTENING

    def __str__(self):
        return "ListeningState"

    def _prepare(self):
        self.bira_manager.prepare_listening()

    def _handle(self):
        transcription = self.bira_manager.controller.listen()
        self.bira_manager.add_user_input(transcription)
        self.bira_manager.get_data().listening_code = ListeningCode.SUCCESS

    def _decide_next_state(self):
        listening_code = self.bira_manager.get_data().listening_code
        feedback = None
        new_state = StateCode.EXIT

        match listening_code:
            case ListeningCode.ERROR:
                self.log_state("Error occurred during listening processing.")
                new_state = StateCode.EXIT
            case ListeningCode.NO_RESPONSE:
                self.log_state("Listening has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case ListeningCode.SUCCESS:
                self.log_state("Listening processing successful.")
                feedback = f"Vous m'avez demandé: {self.bira_manager.get_last_user_input()}."
                # Always transition to Planning; Planning will decide if vision is needed.
                new_state = StateCode.PLANNING
            case ListeningCode.NO_INPUT:
                self.log_state("No voice input received.")
                feedback = "I didn't hear your command. Going back to standby."
                new_state = StateCode.IDLE
            case _:
                self.log_state("Unknown listening code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT

        if feedback:
            self.emit_feedback(feedback, source="state_log")
        self.bira_manager.change_state(new_state)