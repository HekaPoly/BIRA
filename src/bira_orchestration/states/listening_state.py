from bira_orchestration.states.base_state import State
from bira_orchestration.enums import StateCode, ListeningCode, PlanificationCode

class ListeningState(State):
    code = StateCode.LISTENING

    def __str__(self):
        return "ListeningState"

    def _prepare(self):
        self.bira_manager.prepare_listening()

    def _handle(self):
        # TODO: Implement actual listening logic
        transcription = self.bira_manager.controller.listen()
        self.bira_manager.add_user_input(transcription)
        self.bira_manager.get_data().listening_code = ListeningCode.SUCCESS

    def _decide_next_state(self):
        listening_code = self.bira_manager.get_data().listening_code
        context = self.bira_manager.get_data()
        feedback = None
        new_state = StateCode.EXIT

        match listening_code:
            case ListeningCode.ERROR:
                print("Error occurred during listening processing.")
                new_state = StateCode.EXIT
            case ListeningCode.NO_RESPONSE:
                print("Listening has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case ListeningCode.SUCCESS:
                print("Listening processing successful.")
                feedback = f"Vous m'avez demandé: {self.bira_manager.get_last_user_input()}."
                if context.planification_code in {
                    PlanificationCode.REPEAT_REQUEST,
                    PlanificationCode.NEED_MORE_INFO,
                }:
                    new_state = StateCode.PLANNING
                else:
                    new_state = StateCode.VISION
            case ListeningCode.NO_INPUT:
                print("No voice input received.")
                feedback = "Je n'ai pas entendu votre commande. Je vais me remettre en veille."
                new_state = StateCode.IDLE
            case _:
                print("Unknown listening code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT

        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(new_state)