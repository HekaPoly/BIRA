from bira_orchestration.enums import StateCode, ExecutionCode
from bira_orchestration.states.base_state import State


class ExecutingState(State):
    code = StateCode.EXECUTING

    def __str__(self):
        return "ExecutingState"

    def _prepare(self):
        self.bira_manager.prepare_execution()

    def _handle(self):
        self.bira_manager.controller.send_mechanical_command(self.bira_manager.get_data().object_selected)
        self.bira_manager.get_data().execution_code = ExecutionCode.SUCCESS

    def _decide_next_state(self):
        execution_code = self.bira_manager.get_data().execution_code
        feedback = None
        new_state = StateCode.EXIT

        match execution_code:
            case ExecutionCode.ERROR:
                print("Error occurred during execution processing.")
                new_state = StateCode.EXIT
            case ExecutionCode.NO_RESPONSE:
                print("Execution has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case ExecutionCode.SUCCESS:
                print("Execution processing successful.")
                feedback = "J'ai exécuté la tâche demandée. Voulez-vous que je fasse autre chose ?"
                new_state = StateCode.LISTENING
            case ExecutionCode.UNABLE_TO_MOVE:
                print("Unable to move. I might be blocked or there might be an obstacle.")
                feedback = "Je n'ai pas pu atteindre l'objet. Il semble qu'il y ait un obstacle ou que je suis bloqué."
                new_state = StateCode.IDLE
            case ExecutionCode.UNREACHABLE_OBJECT:
                print(
                    "Unreachable object. The object might be out of my reach or might have been moved since the vision stage.")
                feedback = "Je n'ai pas pu atteindre l'objet. Il semble que l'objet soit hors de ma portée ou qu'il ait été déplacé depuis la vision."
                new_state = StateCode.VISION
            case ExecutionCode.OBJECT_DROPPED:
                print("Object dropped. I might have dropped the object during the execution.")
                feedback = "J'ai laissé tomber l'objet pendant l'exécution. Je suis désolé. Voulez-vous que je réessaye ?"
                new_state = StateCode.LISTENING
            case _:
                print("Unknown execution code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT

        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(new_state)
