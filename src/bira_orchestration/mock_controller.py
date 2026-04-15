from types import SimpleNamespace
from time import sleep

from bira_components.SLM_Manager import SLM_Manager
from bira_orchestration.enums import PlanificationCode


class _MockComputerVision:
    def __init__(self):
        # Common COCO labels used in default mock scene.
        self._labels = {
            39: "bottle",
            41: "cup",
            67: "cell phone",
        }

    def warmup(self):
        print("[MOCK:VISION] Warmup skipped.")

    def get_label_name(self, label_id: int) -> str:
        return self._labels.get(int(label_id), f"Unknown_{label_id}")

    @property
    def label_dict(self):
        return self._labels


class _MockDetectedObject:
    def __init__(self, obj_id: int, raw_label: int, confidence: int, position: list[float], bbox_2d: list[list[int]]):
        self.id = obj_id
        self.raw_label = raw_label
        self.confidence = confidence
        self.position = position
        self.bounding_box_2d = bbox_2d


class MockedBiraController:
    def __init__(self, slm_mode: str = "local", slm_debug: bool = False, slm_stream: bool = True, api_key=None):
        print("[MOCK MODE] Hardware/audio components are mocked. SLM remains active.")
        self.camera = None
        self.computer_vision = _MockComputerVision()
        self.uart_transmitter = None
        self.micro = None
        self.text_to_speech = None
        self.speech_to_text = None
        self.slm_manager = SLM_Manager(
            mode=slm_mode,
            debug=slm_debug,
            stream_json=slm_stream,
            api_key=api_key,
        )

    def preload_components(self):
        preload_steps = [
            ("language model", self.slm_manager.preload),
        ]

        for label, preload in preload_steps:
            print(f"Preloading {label}...")
            try:
                preload()
            except Exception as exc:
                print(f"Unable to preload {label}: {exc}")

    def sleep_mode(self):
        try:
            input("[MOCK:WAKE] Press Enter to wake BIRA... ")
        except EOFError:
            # In non-interactive test runs, EOF means no further input is available.
            print("[MOCK:WAKE] EOF received; shutting down mock run.")
            raise SystemExit(0)

    def listen(self):
        try:
            transcription = input("[MOCK:USER_INPUT] ").strip()
        except EOFError:
            # Gracefully terminate when stdin is closed (e.g., piped single-turn tests).
            print("[MOCK:USER_INPUT] EOF received; shutting down mock run.")
            raise SystemExit(0)
        if transcription.lower() in {"q", "quit", "exit"}:
            return "stop"
        return transcription

    def vision(self):
        mock_objects = [
            _MockDetectedObject(
                obj_id=1,
                raw_label=39,
                confidence=91,
                position=[0.35, -0.12, 1.05],
                bbox_2d=[[100, 90], [160, 90], [100, 190], [160, 190]],
            ),
            _MockDetectedObject(
                obj_id=2,
                raw_label=39,
                confidence=88,
                position=[0.12, 0.05, 0.82],
                bbox_2d=[[260, 100], [320, 100], [260, 200], [320, 200]],
            ),
            _MockDetectedObject(
                obj_id=3,
                raw_label=41,
                confidence=86,
                position=[-0.10, 0.03, 0.72],
                bbox_2d=[[360, 120], [420, 120], [360, 200], [420, 200]],
            ),
        ]
        detection_labels = [obj.raw_label for obj in mock_objects]
        return SimpleNamespace(object_list=mock_objects), detection_labels

    def send_mechanical_command(self, command):
        label_name = "unknown"
        raw = getattr(command, "raw_label", None)
        if raw is not None and self.computer_vision is not None:
            label_name = self.computer_vision.get_label_name(int(raw))
        print(f"[MOCK:EXECUTION] Pretending to execute grasp action on: {label_name}")
        sleep(0.3)

    def speak(self, text, source: str = "state_log"):
        source_tag = {
            "state_log": "State log",
            "fallback_feedback": "Fallback feedback",
            "slm_feedback": "SLM_feedback",
        }.get(source, source)
        print(f"\n=== BIRA_FEEDBACK [{source_tag}] ===\n{text}\n=====================\n")

    def prompt_slm(self, context):
        if context.user_inputs:
            # Keep only the latest user utterance here; dialog continuity is handled by SLM history.
            user_input = context.user_inputs[-1]
        else:
            user_input = None
        mode_hint = context.route_mode_hint
        if context.skip_vision_for_current_input and mode_hint in {
            "conversing",
            "out_of_scope",
            "repeat",
            "stop",
            "inappropriate",
            "unclear_action",
        }:
            return self.slm_manager.respond_from_mode_hint(
                mode=mode_hint,
                transcription=user_input or "",
                detections=[],
            )

        detected_objects = context.objects_detected if context.objects_detected else None

        self.slm_manager.set_transcription(user_input or "")
        self.slm_manager.set_detections(
            detection_labels=context.detection_labels,
            detected_objects=detected_objects,
            computer_vision=self.computer_vision,
        )

        return self.slm_manager.run_inference()

    def route_request(self, context):
        user_input = context.user_inputs[-1] if context.user_inputs else ""
        return self.slm_manager.route_request(user_input)

    def destroy(self):
        components = [
            self.camera,
            self.computer_vision,
            self.uart_transmitter,
            self.micro,
            self.text_to_speech,
            self.speech_to_text,
            self.slm_manager,
        ]

        for component in components:
            try:
                if component and hasattr(component, "destroy"):
                    component.destroy()
            except Exception as exc:
                print(f"Failed to destroy {component}: {exc}")
