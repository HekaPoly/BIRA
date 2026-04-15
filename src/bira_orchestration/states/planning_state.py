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
        context = self.bira_manager.get_data()
        context.object_selected = None

        # Decide if vision is needed based on transcription.
        user_input = context.user_inputs[-1] if context.user_inputs else ""
        route = self.bira_manager.controller.route_request(context)
        needs_vision = route.get("needs_vision", True)
        mode_hint = route.get("mode")
        context.skip_vision_for_current_input = not needs_vision
        context.route_mode_hint = mode_hint

        # Run vision inline if needed and not already executed.
        if needs_vision and not context.objects_detected:
            self.log_state(f"Planning: Vision needed for '{user_input[:60]}...'")
            try:
                sl_object, detection_labels = self.bira_manager.controller.vision()
                context.objects_detected = sl_object.object_list
                context.detection_labels = detection_labels
                for obj in context.objects_detected:
                    obj_id = obj.id
                    confidence = obj.confidence
                    position_3d = obj.position
                    bbox_2d = obj.bounding_box_2d
                    self.log_state(
                        f"  Object ID: {obj_id}, Confidence: {confidence}, Position: {position_3d}, BBox: {bbox_2d}"
                    )
                if not context.objects_detected:
                    self.log_state("  No objects detected.")
                else:
                    cv = self.bira_manager.controller.computer_vision
                    label_names = [cv.get_label_name(label_id) for label_id in context.detection_labels]
                    self.log_state(f"  Detected: {', '.join(label_names)}")
            except Exception as exc:
                self.log_state(f"Vision error: {exc}")
                context.objects_detected = []
                context.detection_labels = []
        elif not needs_vision:
            self.log_state(f"Planning: No vision needed for '{user_input[:60]}...' (mode_hint={mode_hint})")
            if mode_hint not in {"clarification", "confirmation"}:
                # Prevent stale detections from previous completed turns from leaking into this turn.
                context.objects_detected = []
                context.detection_labels = []

        # Now prompt SLM with available vision data (or empty if not needed).
        response = self.bira_manager.controller.prompt_slm(context)
        feedback = response.get("response", "I didn't understand the request.")
        mode = response.get("mode", "clarification")
        context.feedback_source = response.get("feedback_source", "slm_feedback")

        # Route based on SLM's mode decision (which has already validated the logic)
        if mode == "stop":
            context.clear_vision_context()
            self.bira_manager.controller.slm_manager.reset_task_context()
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
            self.bira_manager.add_feedback("Error: object not found despite confirmation.")
            context.feedback_source = "fallback_feedback"
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
                self.log_state("Error occurred during planification processing.")
                new_state = StateCode.EXIT
            case PlanificationCode.NO_RESPONSE:
                self.log_state("Planification has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case PlanificationCode.SUCCESS:
                self.log_state("Planification processing successful.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.EXECUTING
            case PlanificationCode.UNCLEAR_COMMAND:
                self.log_state("Unclear command. User needs to reformulate.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.INAPPROPRIATE_REQUEST:
                self.log_state("Inappropriate request. Content is not allowed.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.OUT_OF_SCOPE_REQUEST:
                self.log_state("Out-of-scope action. User should ask for a supported robotic-arm task.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.UNDETECTED_OBJECT:
                self.log_state("Object not detected. Vision needs to be redone.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.VISION
            case PlanificationCode.REPEAT_REQUEST:
                self.log_state("User needs to repeat the command.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.NEED_MORE_INFO:
                self.log_state("More information needed to identify the object.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.REFORMULATE_REQUEST:
                self.log_state("Requested object not found. User needs to reformulate.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.CONVERSING:
                self.log_state("Conversational exchange only. Returning to listening.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.IDLE:
                self.log_state("Idle state reached in planification. Transitioning to RespondingState with feedback.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.IDLE
            case _:
                self.log_state("Unknown planification code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT

        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback, source=self.bira_manager.get_data().feedback_source)
        self.bira_manager.change_state(new_state)
