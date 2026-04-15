from bira_orchestration.states.base_state import State
from bira_orchestration.enums import VisionCode, StateCode

class VisionState(State):
    code = StateCode.VISION

    def __str__(self):
        return "VisionState"

    def _prepare(self):
        self.bira_manager.prepare_vision()

    def _handle(self):
        sl_object, detection_labels = self.bira_manager.controller.vision()
        context = self.bira_manager.get_data()
        context.objects_detected = sl_object.object_list
        context.detection_labels = detection_labels
        context.vision_code = VisionCode.SUCCESS

        for obj in context.objects_detected:
            obj_id = obj.id
            confidence = obj.confidence
            position_3d = obj.position  # [x, y, z]
            bbox_2d = obj.bounding_box_2d
            self.log_state(f"Object ID: {obj_id}, Confidence: {confidence}, Position: {position_3d}, BBox: {bbox_2d}")

        if not context.objects_detected:
            self.log_state("No objects returned by vision.")

        self.log_state(f"Corresponding labels: {context.detection_labels}")

    def _decide_next_state(self):
        vision_code = self.bira_manager.get_data().vision_code
        feedback = None
        new_state = StateCode.EXIT

        match vision_code:
            case VisionCode.ERROR:
                self.log_state("Error occurred during vision processing.")
                new_state = StateCode.EXIT
            case VisionCode.NO_RESPONSE:
                self.log_state("Vision has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case VisionCode.SUCCESS:
                self.log_state("Vision processing successful.")
                detection_labels = self.bira_manager.get_data().detection_labels or []
                cv = self.bira_manager.controller.computer_vision
                label_names = [cv.get_label_name(label_id) for label_id in detection_labels]
                feedback = f"J'ai détecté les objets suivants: {', '.join(label_names)}."
                new_state = StateCode.PLANNING
            case VisionCode.NO_OBJECT_DETECTED:
                self.log_state("No objects detected.")
                feedback = "Je n'ai détecté aucun objet. Veuillez réessayer."
                new_state = StateCode.LISTENING
            case _:
                self.log_state("Unknown vision code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT

        if feedback:
            self.emit_feedback(feedback, source="state_log")
        self.bira_manager.change_state(new_state)
