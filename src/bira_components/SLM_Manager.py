"""Simplified SLM_Manager using Ollama's native capabilities."""

from __future__ import annotations

import json
import os
from typing import Optional

from ollama import Client

from bira_components.slm_controller import SLM_Controller
from bira_components.slm_formatter import SLM_Formatter
from bira_components.tensorrt_manager import TensorRT_Manager


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class SLM_Manager:
    """Simplified SLM_Manager using Ollama's native thinking and streaming."""

    def __init__(
        self,
        model_name: str = "bira-assistant",
        temperature: float = 0.3,
        mode: str = "local",
        api_key: Optional[str] = None,
        tensorrt_engine_dir: Optional[str] = None,
    ):
        """
        Initialize SLM_Manager.

        Args:
            model_name: Ollama model name (default: 'bira-assistant' - the custom Modelfile model)
            temperature: Sampling temperature (0.3 = more focused)
            mode: 'local' or 'cloud'
            api_key: API key for cloud mode
        """
        self.model_name = model_name
        self.temperature = temperature
        self.mode = mode
        self.history = []  # No system prompt in history - it's in the Modelfile
        self.detections = None
        self.detection_candidates = None
        self.transcription = None
        self.debug = _env_flag("SLM_DEBUG", default=False)
        self.stream_json = _env_flag("SLM_STREAM", default=False)
        self.num_predict = int(os.getenv("SLM_NUM_PREDICT", "900"))
        self.pending_label: Optional[str] = None
        # Policy: always prefer TensorRT when available in local mode.
        self.prefer_tensorrt = True
        self.response_schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": [
                        "confirmation",
                        "clarification",
                        "reformulate",
                        "repeat",
                        "unclear_action",
                        "conversing",
                        "out_of_scope",
                        "inappropriate",
                        "stop",
                    ],
                },
                "selected_candidate_index": {
                    "type": ["integer", "null"],
                },
                "selected_label": {"type": ["string", "null"]},
                "selected_label_id": {"type": ["integer", "null"]},
            },
            "required": [
                "response",
                "mode",
                "selected_candidate_index",
                "selected_label",
                "selected_label_id",
            ],
            "additionalProperties": False,
        }

        if mode == "cloud":
            if not api_key:
                raise ValueError("API key required for cloud mode")
            self.client = Client(
                host="https://ollama.com",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        else:
            self.client = Client(host="http://localhost:11434")

        self.formatter = SLM_Formatter(response_schema=self.response_schema)
        self.chat_controller = SLM_Controller(
            client=self.client,
            model_name=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            debug=self.debug,
        )
        self.tensorrt_manager = TensorRT_Manager(
            mode=self.mode,
            prefer_tensorrt=self.prefer_tensorrt,
            engine_dir=tensorrt_engine_dir,
        )
        self.tensorrt_manager.initialize()
        self.prefer_tensorrt = self.tensorrt_manager.prefer_tensorrt

    def _debug_log(self, message: str) -> None:
        """Log debug message if SLM_DEBUG is enabled."""
        if self.debug:
            print(f"[SLM_DEBUG] {message}")

    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.history = []
        self.detections = None
        self.detection_candidates = None
        self.transcription = None
        self.pending_label = None

    def set_transcription(self, transcription: str) -> None:
        """Set user transcription."""
        self.transcription = transcription

    def set_detections(
        self,
        detection_labels: Optional[list[int]] = None,
        detected_objects: Optional[list] = None,
        computer_vision=None,
    ) -> None:
        """
        Set detected objects and resolve labels to human-readable names.

        Args:
            detection_labels: List of YOLO label IDs
            detected_objects: List of detected objects with raw_label attribute
            computer_vision: ComputerVision instance for label name resolution
        """
        resolved: list[str] = []
        candidates: list[dict] = []

        # Prefer detected_objects if available (has position data)
        if detected_objects and computer_vision:
            for index, obj in enumerate(detected_objects):
                raw = getattr(obj, "raw_label", None)
                if raw is None:
                    continue
                label_id = int(raw)
                label_name = computer_vision.get_label_name(label_id)
                resolved.append(label_name)

                position = getattr(obj, "position", None)
                pos_value = (
                    [float(position[0]), float(position[1]), float(position[2])]
                    if position and len(position) >= 3
                    else None
                )

                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": label_id,
                        "position": pos_value,
                    }
                )
        # Fallback to detection_labels if objects unavailable
        elif detection_labels and computer_vision:
            for index, label_id in enumerate(detection_labels):
                label_name = computer_vision.get_label_name(label_id)
                resolved.append(label_name)
                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": int(label_id),
                        "position": None,
                    }
                )
        elif detection_labels:
            # Standalone usage without ComputerVision
            for index, label_id in enumerate(detection_labels):
                label_name = str(label_id)
                resolved.append(label_name)
                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": int(label_id),
                        "position": None,
                    }
                )

        # Keep duplicates to accurately reflect object counts
        self.detections = resolved if resolved else None
        self.detection_candidates = candidates if candidates else None

    def route_request(self, transcription: str) -> dict:
        """Decide whether vision is needed before planning."""
        text = str(transcription or "").strip().lower()
        if not text:
            return {"needs_vision": False, "mode": "repeat"}

        stop_markers = {
            "stop",
            "cancel",
            "quit",
            "exit",
            "nevermind",
            "never mind",
        }
        if text in stop_markers:
            return {"needs_vision": False, "mode": "stop"}

        out_of_scope_actions = {
            "walk",
            "run",
            "dance",
            "climb",
            "fly",
            "swim",
            "cook",
            "drive",
            "call",
            "sing",
            "open",
            "close",
            "pet",
            "hug",
            "jump",
            "skip",
            "ride",
        }
        in_scope_actions = {
            "pick",
            "grab",
            "bring",
            "fetch",
            "show",
            "give",
            "hold",
            "take",
            "move",
        }
        chat_markers = {
            "how are you",
            "hello",
            "hi",
            "hey",
            "good morning",
            "good evening",
            "what's your name",
            "who are you",
            "thank you",
            "thanks",
            "do you like",
        }

        if any(token in text for token in out_of_scope_actions):
            return {"needs_vision": False, "mode": "out_of_scope"}
        if any(token in text for token in in_scope_actions):
            return {"needs_vision": True, "mode": "clarification"}
        if any(token in text for token in chat_markers):
            return {"needs_vision": False, "mode": "conversing"}

        return {"needs_vision": True, "mode": "clarification"}

    def _parse_response(self, raw_response: str) -> dict:
        """Parse model response JSON through formatter."""
        return self.formatter.parse_response(raw_response)

    def _find_requested_label(self, transcription: str, candidates: list[dict]) -> Optional[str]:
        return self.formatter.find_requested_label(transcription, candidates)

    @staticmethod
    def _is_disambiguation_followup(transcription: str) -> bool:
        return SLM_Formatter.is_disambiguation_followup(transcription)

    def _select_active_candidates(self, transcription: str, candidates: list[dict]) -> list[dict]:
        return self.formatter.select_active_candidates(
            transcription=transcription,
            candidates=candidates,
            pending_label=self.pending_label,
        )

    def _sanitize_output(
        self,
        parsed: dict,
        transcription: str,
        candidates: list[dict],
    ) -> dict:
        return self.formatter.sanitize_output(
            parsed=parsed,
            transcription=transcription,
            candidates=candidates,
            pending_label=self.pending_label,
        )

    def _fallback_mode_response(self, mode: str, transcription: str, detections: list[str]) -> str:
        if mode == "conversing":
            return "I'm doing well, thanks for asking!"
        if mode == "out_of_scope":
            seen = ", ".join(detections) if detections else "no objects right now"
            return (
                "I'm sorry, I can't do that action. "
                f"I'm a robotic arm and can pick, grab, or bring objects. I can see: {seen}."
            )
        if mode == "repeat":
            return "I didn't catch that. Could you repeat?"
        if mode == "stop":
            return "Alright, cancelling."
        if mode == "inappropriate":
            return "I can't help with that. Please ask something safe."
        if mode == "unclear_action":
            return "I couldn't understand the action. Could you ask me to pick, grab, or bring an object?"
        return "Could you rephrase your request?"

    def respond_from_mode_hint(self, mode: str, transcription: str, detections: Optional[list[str]] = None) -> dict:
        detections = detections or []
        system_prompt = (
            "You are a concise robotic-arm assistant. "
            "Write exactly one short, natural sentence for the user based on the provided mode."
        )
        user_prompt = (
            f"Mode: {mode}\\n"
            f"User transcription: {transcription}\\n"
            f"Detected objects: {detections}\\n"
            "Rules: never output JSON, never output mode name, be polite and practical."
        )

        try:
            text = self.chat_controller.chat_non_stream_text(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=100,
            )
        except Exception as exc:
            self._debug_log(f"mode response fallback due to error: {exc}")
            text = ""

        if not text:
            text = self._fallback_mode_response(mode=mode, transcription=transcription, detections=detections)

        if mode == "stop":
            self.reset_conversation()

        self.pending_label = None
        return {
            "response": text,
            "mode": mode,
            "selected_label": None,
            "selected_label_id": None,
            "selected_candidate_index": None,
        }

    def run_inference(self) -> dict:
        """Run inference on current transcription and detections."""
        transcription = self.transcription
        detections = self.detections
        candidates = self.detection_candidates

        # Early exit for missing transcription
        if not transcription:
            return {
                "response": "I didn't hear a command. Could you repeat?",
                "mode": "repeat",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }

        detections = detections or []
        candidates = candidates or []

        active_candidates = self._select_active_candidates(transcription, candidates)

        # Build prompt and add to history
        prompt = self.formatter.build_prompt(
            transcription=transcription,
            detections=detections,
            candidates=active_candidates,
            pending_label=self.pending_label,
        )
        self._debug_log(
            f"prompt_build transcription={repr(transcription[:80])} "
            f"detections={len(detections)} candidates={len(candidates)}"
        )

        self.history.append({"role": "user", "content": prompt})

        # Prefer TensorRT when available; fallback to Ollama streaming.
        try:
            thinking = ""
            content = ""

            if self.prefer_tensorrt and self.tensorrt_manager.ready:
                try:
                    content = self.tensorrt_manager.chat(
                        messages=self.history,
                        max_new_tokens=self.num_predict,
                        temperature=self.temperature,
                    ) or ""
                    self._debug_log(f"tensorrt_response_len={len(content)}")
                except Exception as exc:
                    self._debug_log(f"tensorrt_error={exc}")
                    # Runtime TensorRT failures should not silently fallback when TensorRT is detected.
                    raise

            # If TensorRT is unavailable or returned empty, use Ollama JSON chat.
            if not content.strip():
                if self.stream_json:
                    thinking, content = self.chat_controller.chat_stream_json(
                        messages=self.history,
                        schema=self.response_schema,
                    )
                else:
                    thinking, content = self.chat_controller.chat_non_stream_json(
                        messages=self.history,
                        schema=self.response_schema,
                    )
                    if self.debug and thinking.strip():
                        print("[SLM_DEBUG] Thinking:")
                        print(thinking)
                    if self.debug and content.strip():
                        print("[SLM_DEBUG] Answer:")
                        print(content)

            if self.debug and (thinking or content):
                print("")

            self._debug_log(f"response_len={len(content)}")

            # Fallback if stream produced thinking but no answer tokens.
            if not content.strip():
                self._debug_log("empty stream content, retrying non-stream without thinking")
                _, retry_content = self.chat_controller.chat_non_stream_json(
                    messages=self.history,
                    schema=self.response_schema,
                )
                content = retry_content or content

            # Add assistant response to history for dialog continuity
            if content.strip():
                assistant_message = {"role": "assistant", "content": content}
                if thinking.strip():
                    assistant_message["thinking"] = thinking
                self.history.append(assistant_message)

                # Parse and validate response
                parsed = self._parse_response(content)
                parsed = self._sanitize_output(
                    parsed=parsed,
                    transcription=transcription,
                    candidates=active_candidates,
                )

                # Reset on stop
                if parsed["mode"] == "stop":
                    self.reset_conversation()

                if parsed["mode"] == "clarification":
                    requested_label = self._find_requested_label(transcription, active_candidates)
                    self.pending_label = requested_label or self.pending_label
                else:
                    # Only keep pending label across clarification turns.
                    self.pending_label = None

                return parsed
            else:
                self._debug_log("WARNING: empty response from model")
                return {
                    "response": "I didn't understand. Could you repeat?",
                    "mode": "clarification",
                    "selected_label": None,
                    "selected_label_id": None,
                    "selected_candidate_index": None,
                }

        except json.JSONDecodeError as err:
            self._debug_log(f"JSON parse error: {err}")
            return {
                "response": "I had trouble processing that. Could you repeat?",
                "mode": "clarification",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }
        except Exception as err:
            self._debug_log(f"inference error: {err}")
            raise

    def load_model(self) -> None:
        """Check that model is available (simple validation)."""
        if self.prefer_tensorrt and self.tensorrt_manager.ready:
            try:
                self.tensorrt_manager.warmup()
                self._debug_log("TensorRT backend is ready")
                return
            except Exception as exc:
                self._debug_log(f"TensorRT warmup failed: {exc}")
                # TensorRT detected but failed warmup: surface the failure.
                raise RuntimeError(f"TensorRT warmup failed: {exc}") from exc

        try:
            # Quick warmup call
            result = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "{}"}],
                options={"num_predict": 2},
            )
            self._debug_log(f"Model {self.model_name} is ready")
        except Exception as exc:
            raise RuntimeError(
                f"Model '{self.model_name}' not found or not ready. "
                f"Create it with: ollama create {self.model_name} -f Modelfile"
            ) from exc

    def preload(self) -> None:
        """Preload model."""
        self.load_model()


if __name__ == "__main__":
    # Quick test
    os.environ["SLM_DEBUG"] = "1"
    mgr = SLM_Manager()
    mgr.load_model()
    mgr.set_transcription("give me a bottle")
    mgr.set_detections(detection_labels=[0, 0, 1])  # 2 bottles, 1 cup
    result = mgr.run_inference()
    print(json.dumps(result, indent=2))
