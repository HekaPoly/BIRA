from bira_orchestration.states.base_state import State
from bira_orchestration.enums import VisionCode, StateCode

class VisionState(State):
    code = StateCode.VISION

    def __str__(self):
        return "VisionState"

    def _prepare(self):
        self.bira_manager.prepare_vision()

    def _handle(self):
        # TODO: Implement actual vision logic
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
            print(f"Object ID: {obj_id}, Confidence: {confidence}, Position: {position_3d}, BBox: {bbox_2d}")

        if not context.objects_detected:
            print("No objects returned by vision.")

        print(f"Corresponding labels: {context.detection_labels}")

    def _decide_next_state(self):
        vision_code = self.bira_manager.get_data().vision_code
        feedback = None
        new_state = StateCode.EXIT

        match vision_code:
            case VisionCode.ERROR:
                print("Error occurred during vision processing.")
                new_state = StateCode.EXIT
            case VisionCode.NO_RESPONSE:
                print("Vision has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case VisionCode.SUCCESS:
                print("Vision processing successful.")
                feedback = f"J'ai détecté les objets suivants: {', '.join(str(self.bira_manager.get_data().detection_labels))}."
                new_state = StateCode.PLANNING
            case VisionCode.NO_OBJECT_DETECTED:
                print("No objects detected.")
                feedback = "Je n'ai détecté aucun objet. Veuillez réessayer."
                new_state = StateCode.LISTENING
            case _:
                print("Unknown vision code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT

        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(new_state)
