"""Simplified SLM_Manager using Ollama's native capabilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ollama import Client

try:
    from bira_components.tensorRT import TensorRTInferenceEngine
except Exception:
    TensorRTInferenceEngine = None


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_first_json(text: str) -> dict:
    """Parse the first complete JSON object from text."""
    text = text.strip()
    object_start = text.find("{")
    array_start = text.find("[")

    if object_start == -1 and array_start == -1:
        raise json.JSONDecodeError("No JSON value found", text, 0)

    start = (
        object_start
        if array_start == -1
        else array_start
        if object_start == -1
        else min(object_start, array_start)
    )

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise json.JSONDecodeError("Unbalanced JSON", text, start)


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
        self.num_predict = int(os.getenv("SLM_NUM_PREDICT", "900"))
        self.pending_label: Optional[str] = None
        # Policy: always prefer TensorRT when available in local mode.
        self.prefer_tensorrt = True
        default_engine_dir = Path(__file__).resolve().parent / "tensorRT" / "tensorrt_models" / "engines"
        self.tensorrt_engine_dir = Path(tensorrt_engine_dir or default_engine_dir)
        self.trt_engine = None
        self.trt_ready = False
        self.response_schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["confirmation", "clarification", "repeat", "stop"],
                },
                "request_scope": {
                    "type": "string",
                    "enum": ["in_scope", "out_of_scope"],
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
                "request_scope",
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

        if self.mode != "local" and self.prefer_tensorrt:
            print("TensorRT is only available in local mode. Falling back to Ollama.")
            self.prefer_tensorrt = False

        if self.prefer_tensorrt:
            if TensorRTInferenceEngine is None:
                print("TensorRT support is unavailable. Falling back to Ollama.")
                self.prefer_tensorrt = False
            else:
                self._init_tensorrt_engine()

    def _debug_log(self, message: str) -> None:
        """Log debug message if SLM_DEBUG is enabled."""
        if self.debug:
            print(f"[SLM_DEBUG] {message}")

    def _init_tensorrt_engine(self) -> None:
        try:
            self.trt_engine = TensorRTInferenceEngine(engine_dir=str(self.tensorrt_engine_dir))
            self.trt_ready = bool(self.trt_engine.load_engine())

            if self.trt_ready:
                print(f"TensorRT ready (engine: {self.tensorrt_engine_dir})")
                return

            self.trt_engine = None
            # If TensorRT is not detected/ready, fallback to Ollama.
            print(f"TensorRT engine not ready in {self.tensorrt_engine_dir}. Falling back to Ollama.")
        except Exception as exc:
            self.trt_ready = False
            self.trt_engine = None
            print(f"TensorRT unavailable ({exc}). Falling back to Ollama.")

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

    def _build_prompt(self, transcription: str, detections: list[str], candidates: Optional[list[dict]]) -> str:
        """Build user prompt with detections and candidates."""
        candidates_json = json.dumps(candidates or [], ensure_ascii=False)
        schema_json = json.dumps(self.response_schema, ensure_ascii=False)
        pending_label_text = self.pending_label or "null"
        return (
            f"Transcription: {transcription}\n"
            f"Detected objects: {detections}\n"
            f"Pending label from prior clarification (if any): {pending_label_text}\n"
            f"Candidates (format: index, label, label_id, position [x,y,z]): {candidates_json}\n"
            "Important: when transcription references prior clarification (e.g., 'left one', 'closest one'), "
            "resolve using candidates positions and return confirmation with selected_candidate_index.\n"
            "Important: if pending label exists and user says pronouns like 'left one', resolve only within that label group.\n"
            f"Return JSON matching this schema exactly: {schema_json}"
        )

    @staticmethod
    def _as_dict(value):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return {}

    def _parse_response(self, raw_response: str) -> dict:
        """Parse model response JSON."""
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = _parse_first_json(raw_response)

        # Handle array responses (take last element)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[-1]

        if not isinstance(parsed, dict):
            raise ValueError("Response is not JSON object")

        # Validate and normalize fields
        mode = str(parsed.get("mode", "clarification")).strip().lower()
        if mode not in {"confirmation", "clarification", "repeat", "stop"}:
            mode = "clarification"

        request_scope = str(parsed.get("request_scope", "in_scope")).strip().lower()
        if request_scope not in {"in_scope", "out_of_scope"}:
            request_scope = "in_scope"

        response_text = str(parsed.get("response", "I didn't understand. Could you repeat?"))

        selected_label = parsed.get("selected_label")
        selected_label = (str(selected_label).strip() or None) if selected_label is not None else None

        selected_label_id = parsed.get("selected_label_id")
        try:
            selected_label_id = int(selected_label_id) if selected_label_id is not None else None
        except (TypeError, ValueError):
            selected_label_id = None

        selected_candidate_index = parsed.get("selected_candidate_index")
        try:
            selected_candidate_index = int(selected_candidate_index) if selected_candidate_index is not None else None
        except (TypeError, ValueError):
            selected_candidate_index = None

        return {
            "response": response_text,
            "mode": mode,
            "request_scope": request_scope,
            "selected_label": selected_label,
            "selected_label_id": selected_label_id,
            "selected_candidate_index": selected_candidate_index,
        }

    @staticmethod
    def _pluralize(label: str, count: int) -> str:
        if count == 1:
            return label
        if label.endswith("s"):
            return label
        return f"{label}s"

    def _find_requested_label(self, transcription: str, candidates: list[dict]) -> Optional[str]:
        text = str(transcription or "").strip().lower()
        if not text or not candidates:
            return None

        label_counts: dict[str, int] = {}
        for candidate in candidates:
            label = str(candidate.get("label") or "").strip().lower()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        for label in label_counts:
            if label in text:
                return label

        return None

    @staticmethod
    def _is_disambiguation_followup(transcription: str) -> bool:
        text = str(transcription or "").strip().lower()
        if not text:
            return False
        hints = [
            "left",
            "right",
            "top",
            "bottom",
            "closest",
            "farthest",
            "one",
            "this",
            "that",
        ]
        return any(hint in text for hint in hints)

    def _select_active_candidates(self, transcription: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        explicit_label = self._find_requested_label(transcription, candidates)
        if explicit_label:
            return [c for c in candidates if str(c.get("label") or "").strip().lower() == explicit_label]

        if self.pending_label and self._is_disambiguation_followup(transcription):
            narrowed = [
                c for c in candidates if str(c.get("label") or "").strip().lower() == self.pending_label
            ]
            if narrowed:
                return narrowed

        return candidates

    def _sanitize_output(
        self,
        parsed: dict,
        transcription: str,
        candidates: list[dict],
    ) -> dict:
        response_text = str(parsed.get("response") or "").strip()
        normalized = response_text.lower()
        requested_label = self._find_requested_label(transcription, candidates) or self.pending_label

        if not response_text or normalized in {"...", "…"}:
            if parsed.get("mode") == "confirmation":
                parsed["response"] = "Understood. I will pick that object."
            else:
                if requested_label:
                    count = sum(
                        1 for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
                    )
                    if count > 1:
                        parsed["response"] = (
                            f"I found {count} {self._pluralize(requested_label, count)}. "
                            "Which one would you like? (e.g., left, right, closest, farthest?)"
                        )
                        parsed["mode"] = "clarification"
                    elif count == 1:
                        only = next(
                            c for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
                        )
                        parsed["response"] = f"I found one {requested_label}. I can pick it now."
                        parsed["mode"] = "confirmation"
                        parsed["selected_candidate_index"] = only.get("index")
                        parsed["selected_label"] = only.get("label")
                        parsed["selected_label_id"] = only.get("label_id")
                    else:
                        parsed["response"] = "I need more details to identify the object."
                        parsed["mode"] = "clarification"
                else:
                    parsed["response"] = "I need more details to identify the object."
                    parsed["mode"] = "clarification"

        if parsed.get("mode") == "clarification" and (
            "don't see" in normalized or "do not see" in normalized
        ) and requested_label:
            count = sum(
                1 for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
            )
            if count > 1:
                parsed["response"] = (
                    f"I found {count} {self._pluralize(requested_label, count)}. "
                    "Which one would you like? (e.g., left, right, closest, farthest?)"
                )
            elif count == 1:
                only = next(
                    c for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
                )
                parsed["response"] = f"I found one {requested_label}. I can pick it now."
                parsed["mode"] = "confirmation"
                parsed["selected_candidate_index"] = only.get("index")
                parsed["selected_label"] = only.get("label")
                parsed["selected_label_id"] = only.get("label_id")

        if parsed.get("mode") == "confirmation" and parsed.get("selected_candidate_index") is None:
            if requested_label:
                same = [
                    c for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
                ]
                if len(same) == 1:
                    parsed["selected_candidate_index"] = same[0].get("index")
                    parsed["selected_label"] = same[0].get("label")
                    parsed["selected_label_id"] = same[0].get("label_id")
                elif len(same) > 1:
                    parsed["mode"] = "clarification"
                    parsed["response"] = (
                        f"I found {len(same)} {self._pluralize(requested_label, len(same))}. "
                        "Which one would you like? (e.g., left, right, closest, farthest?)"
                    )
                    parsed["selected_label"] = None
                    parsed["selected_label_id"] = None
        return parsed

    def _chat_non_stream(self) -> tuple[str, str]:
        fallback = self.client.chat(
            model=self.model_name,
            messages=self.history,
            stream=False,
            think=False,
            format=self.response_schema,
            options={"temperature": self.temperature, "num_predict": self.num_predict},
        )
        data = self._as_dict(fallback)
        message = self._as_dict(data.get("message", {}))
        return str(message.get("thinking") or ""), str(message.get("content") or "")

    def _chat_tensorrt(self) -> tuple[str, str]:
        if not (self.prefer_tensorrt and self.trt_ready and self.trt_engine):
            return "", ""

        response = self.trt_engine.chat(
            messages=self.history,
            max_new_tokens=self.num_predict,
            temperature=self.temperature,
        )
        data = self._as_dict(response)
        message = self._as_dict(data.get("message", {}))
        return "", str(message.get("content") or "")

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
                "request_scope": "in_scope",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }

        # Early exit for missing detections
        if not detections:
            return {
                "response": "I don't see any relevant object. Could you clarify?",
                "mode": "clarification",
                "request_scope": "in_scope",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }

        active_candidates = self._select_active_candidates(transcription, candidates or [])

        # Build prompt and add to history
        prompt = self._build_prompt(transcription, detections, active_candidates)
        self._debug_log(
            f"prompt_build transcription={repr(transcription[:80])} "
            f"detections={len(detections)} candidates={len(candidates or [])}"
        )

        self.history.append({"role": "user", "content": prompt})

        # Prefer TensorRT when available; fallback to Ollama streaming.
        try:
            thinking = ""
            content = ""

            if self.prefer_tensorrt and self.trt_ready and self.trt_engine:
                try:
                    _, trt_content = self._chat_tensorrt()
                    content = trt_content or ""
                    self._debug_log(f"tensorrt_response_len={len(content)}")
                except Exception as exc:
                    self._debug_log(f"tensorrt_error={exc}")
                    # Runtime TensorRT failures should not silently fallback when TensorRT is detected.
                    raise

            # If TensorRT is unavailable or returned empty, use Ollama stream.
            if not content.strip():
                stream = self.client.chat(
                    model=self.model_name,
                    messages=self.history,
                    stream=True,
                    think=True,
                    format=self.response_schema,
                    options={"temperature": self.temperature, "num_predict": self.num_predict},
                )

                in_thinking = False

                for chunk in stream:
                    chunk_data = self._as_dict(chunk)
                    message = self._as_dict(chunk_data.get("message", {}))
                    chunk_thinking = str(message.get("thinking") or "")
                    chunk_content = str(message.get("content") or "")

                    if chunk_thinking:
                        if self.debug and not in_thinking:
                            print("[SLM_DEBUG] Thinking:")
                            in_thinking = True
                        if self.debug:
                            print(chunk_thinking, end="", flush=True)
                        thinking += chunk_thinking
                    elif chunk_content:
                        if self.debug and in_thinking:
                            print("\n[SLM_DEBUG] Answer:")
                            in_thinking = False
                        if self.debug:
                            print(chunk_content, end="", flush=True)
                        content += chunk_content

            if self.debug and (thinking or content):
                print("")

            self._debug_log(f"response_len={len(content)}")

            # Fallback if stream produced thinking but no answer tokens.
            if not content.strip():
                self._debug_log("empty stream content, retrying non-stream without thinking")
                _, retry_content = self._chat_non_stream()
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

                # Guardrail: prevent repeat loops if we have valid transcription and detections
                if (
                    parsed["mode"] == "repeat"
                    and transcription.strip()
                    and detections
                ):
                    parsed["mode"] = "clarification"
                    if "repeat" in str(parsed.get("response", "")).lower():
                        parsed["response"] = (
                            "I understood your request, but need more details. "
                            "Which object do you mean?"
                        )

                # Reset on stop
                if parsed["mode"] == "stop":
                    self.reset_conversation()

                if parsed["mode"] in {"stop", "confirmation"}:
                    self.pending_label = None
                elif parsed["mode"] == "clarification":
                    requested_label = self._find_requested_label(transcription, active_candidates)
                    self.pending_label = requested_label or self.pending_label

                return parsed
            else:
                self._debug_log("WARNING: empty response from model")
                return {
                    "response": "I didn't understand. Could you repeat?",
                    "mode": "clarification",
                    "request_scope": "in_scope",
                    "selected_label": None,
                    "selected_label_id": None,
                    "selected_candidate_index": None,
                }

        except json.JSONDecodeError as err:
            self._debug_log(f"JSON parse error: {err}")
            return {
                "response": "I had trouble processing that. Could you repeat?",
                "mode": "clarification",
                "request_scope": "in_scope",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }
        except Exception as err:
            self._debug_log(f"inference error: {err}")
            raise

    def load_model(self) -> None:
        """Check that model is available (simple validation)."""
        if self.prefer_tensorrt and self.trt_ready and self.trt_engine:
            try:
                self.trt_engine.chat(
                    messages=[{"role": "user", "content": "Reply with {}."}],
                    max_new_tokens=8,
                    temperature=0.0,
                )
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
