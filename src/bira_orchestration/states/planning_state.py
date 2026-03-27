from bira_orchestration.enums import StateCode, PlanificationCode
from bira_orchestration.states.base_state import State


class PlanningState(State):
    code = StateCode.PLANNING

    def __str__(self):
        return "PlanningState"

    def _prepare(self):
        self.bira_manager.prepare_planning()

    def _handle(self):
        # TODO: Perform actions specific to Planning state
        # For example, analyze data from VisionState and make decisions
        # If successful, set feedback and plan next actions
        context = self.bira_manager.get_data()
        response = self.bira_manager.controller.prompt_slm(context)
        feedback = response.get("response", "Je n'ai pas compris la demande.")
        mode = response.get("mode", "clarification")

        self.bira_manager.add_feedback(feedback)

        if mode == "confirmation":
            context.object_selected = context.objects_detected[0] if context.objects_detected else None
            print(context.object_selected)
            context.planification_code = PlanificationCode.SUCCESS
        elif mode == "stop":
            context.planification_code = PlanificationCode.IDLE
        else:
            context.planification_code = PlanificationCode.UNCLEAR_COMMAND

    def _decide_next_state(self):
        planification_code = self.bira_manager.get_data().planification_code
        feedback = None
        new_state = StateCode.EXIT

        match planification_code:
            case PlanificationCode.ERROR:
                print("Error occurred during planification processing.")
                new_state = StateCode.EXIT
            case PlanificationCode.NO_RESPONSE:
                print("Planification has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case PlanificationCode.SUCCESS:
                print("Planification processing successful.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.EXECUTING
            case PlanificationCode.UNCLEAR_COMMAND:
                print("Unclear command. User needs to reformulate.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.INAPPROPRIATE_REQUEST:
                print("Inappropriate request. User needs to ask for a valid object.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.UNDETECTED_OBJECT:
                print("Object not detected. Vision needs to be redone.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.VISION
            case PlanificationCode.IDLE:
                print("Idle state reached in planification. Transitioning to RespondingState with feedback.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.IDLE
            case _:
                print("Unknown planification code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT

        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(new_state)
