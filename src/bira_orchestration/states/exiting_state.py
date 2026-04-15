from bira_orchestration.enums import StateCode
from bira_orchestration.states.base_state import State


class ExitState(State):
    code = StateCode.EXIT

    def __str__(self):
        return "ExitState"

    def _prepare(self):
        # Keep the information about the error that caused the exit in the feedback, to be able to send it to the
        # user or save it for later analysis
        pass

    def _handle(self):
        # Perform actions specific to Exit state
        # For example, clean up resources, save state, or perform any necessary shutdown procedures
        feedback = "Une erreur est survenue. Je vais devoir m'arrêter. Veuillez vérifier le système et réessayer."
        self.emit_feedback(feedback, source="state_log")
        self.bira_manager.controller.destroy()

    def _decide_next_state(self):
        # No next state to transition to since this is the exit state
        self.log_state("Exiting the system. Cleaning up resources and shutting down.")
        exit(0)