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
        self.bira_manager.get_data().set_execution_code(ExecutionCode.SUCCESS)

    def _decide_next_state(self):
        execution_code = self.bira_manager.get_data().execution_code
        feedback = None
        new_state = StateCode.EXIT

        match execution_code:
            case ExecutionCode.ERROR:
                self.log_state("Error occurred during execution processing.")
                new_state = StateCode.EXIT
            case ExecutionCode.NO_RESPONSE:
                self.log_state("Execution has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case ExecutionCode.SUCCESS:
                self.log_state("Execution processing successful.")
                feedback = "I completed the requested task. Would you like me to do something else?"
                self.bira_manager.get_data().clear_vision_context()
                self.bira_manager.controller.slm_manager.reset_task_context()
                new_state = StateCode.LISTENING
            case ExecutionCode.UNABLE_TO_MOVE:
                self.log_state("Unable to move. I might be blocked or there might be an obstacle.")
                feedback = "I couldn't reach the object. It seems there's an obstacle or I'm blocked."
                self.bira_manager.get_data().clear_vision_context()
                self.bira_manager.controller.slm_manager.reset_task_context()
                new_state = StateCode.IDLE
            case ExecutionCode.UNREACHABLE_OBJECT:
                self.log_state(
                    "Unreachable object. The object might be out of my reach or might have been moved since the vision stage.")
                feedback = "I couldn't reach the object. It seems the object is out of reach or has moved since detection."
                new_state = StateCode.VISION
            case ExecutionCode.OBJECT_DROPPED:
                self.log_state("Object dropped. I might have dropped the object during the execution.")
                feedback = "I dropped the object during execution. I'm sorry. Would you like me to try again?"
                self.bira_manager.get_data().clear_vision_context()
                self.bira_manager.controller.slm_manager.reset_task_context()
                new_state = StateCode.LISTENING
            case _:
                self.log_state("Unknown execution code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT

        if feedback:
            self.emit_feedback(feedback, source="state_log")
        self.bira_manager.change_state(new_state)
