from bira_orchestration.enums import StateCode, PlanificationCode
from bira_orchestration.states.base_state import State


class PlanningState(State):
    code = StateCode.PLANNING

    def __str__(self):
        return "PlanningState"

    def _prepare(self):
        self.bira_manager.prepare_planning()

    def _resolve_selection(self, context, response):
        selected_candidate_index = response.get("selected_candidate_index")
        selected_label = response.get("selected_label")
        selected_label_id = response.get("selected_label_id")

        # 1) Safest path: explicit candidate index from SLM.
        if selected_candidate_index is not None:
            if 0 <= selected_candidate_index < len(context.objects_detected):
                return [context.objects_detected[selected_candidate_index]]
            return []

        # 2) Fallback path: label id or label name.
        cv = self.bira_manager.controller.computer_vision
        matched_objects = []

        if selected_label_id is not None:
            for obj in context.objects_detected:
                raw = getattr(obj, "raw_label", None)
                if raw is None:
                    continue
                if int(raw) == selected_label_id:
                    matched_objects.append(obj)
            return matched_objects

        if selected_label:
            selected_label_normalized = str(selected_label).strip().lower()
            for obj in context.objects_detected:
                raw = getattr(obj, "raw_label", None)
                if raw is None:
                    continue
                label_name = cv.get_label_name(int(raw)).strip().lower()
                if label_name == selected_label_normalized:
                    matched_objects.append(obj)
            return matched_objects

        return []

    def _handle(self):
        # Single-pass decision making: trust the SLM mode decision and route accordingly.
        context = self.bira_manager.get_data()
        context.object_selected = None

        response = self.bira_manager.controller.prompt_slm(context)
        feedback = response.get("response", "Je n'ai pas compris la demande.")
        mode = response.get("mode", "clarification")

        # Route based on SLM's mode decision (which has already validated the logic)
        if mode == "stop":
            self.bira_manager.add_feedback(feedback)
            context.planification_code = PlanificationCode.IDLE
            return

        if mode == "confirmation":
            matched_objects = self._resolve_selection(context, response)
            if matched_objects:  # Safety check: confirm object exists
                selected_candidate_index = response.get("selected_candidate_index")
                context.object_selected = matched_objects[0]
                self.bira_manager.add_feedback(feedback)
                context.planification_code = PlanificationCode.SUCCESS
                return
            # Fallback (shouldn't happen if SLM is correct)
            self.bira_manager.add_feedback("Erreur: objet non trouvé malgré la confirmation.")
            context.planification_code = PlanificationCode.UNDETECTED_OBJECT
            return

        if mode == "reformulate":
            # SLM found no matching candidate for requested object.
            self.bira_manager.add_feedback(feedback)
            context.planification_code = PlanificationCode.REFORMULATE_REQUEST
            return

        if mode == "unclear_action":
            # SLM understood speech but not the requested action intent.
            self.bira_manager.add_feedback(feedback)
            context.planification_code = PlanificationCode.UNCLEAR_COMMAND
            return

        if mode == "inappropriate":
            # SLM flagged moderated illicit content.
            self.bira_manager.add_feedback(feedback)
            context.planification_code = PlanificationCode.INAPPROPRIATE_REQUEST
            return

        if mode == "out_of_scope":
            # SLM identified an action outside robotic arm capabilities.
            self.bira_manager.add_feedback(feedback)
            context.planification_code = PlanificationCode.OUT_OF_SCOPE_REQUEST
            return

        if mode == "conversing":
            # SLM is having a regular conversation, with no execution requested.
            self.bira_manager.add_feedback(feedback)
            context.planification_code = PlanificationCode.CONVERSING
            return

        if mode == "repeat":
            # SLM couldn't understand input or action
            self.bira_manager.add_feedback(feedback)
            context.planification_code = PlanificationCode.REPEAT_REQUEST
            return

        # mode == "clarification" or any other mode: SLM needs more information
        self.bira_manager.add_feedback(feedback)
        context.planification_code = PlanificationCode.NEED_MORE_INFO

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
                print("Inappropriate request. Content is not allowed.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.OUT_OF_SCOPE_REQUEST:
                print("Out-of-scope action. User should ask for a supported robotic-arm task.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.UNDETECTED_OBJECT:
                print("Object not detected. Vision needs to be redone.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.VISION
            case PlanificationCode.REPEAT_REQUEST:
                print("User needs to repeat the command.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.NEED_MORE_INFO:
                print("More information needed to identify the object.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.REFORMULATE_REQUEST:
                print("Requested object not found. User needs to reformulate.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.CONVERSING:
                print("Conversational exchange only. Returning to listening.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
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
