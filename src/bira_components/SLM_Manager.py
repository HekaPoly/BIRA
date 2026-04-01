from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import argparse
import json
import os
import subprocess

from ollama import Client

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from bira_components.tensorRT import TensorRTInferenceEngine
except Exception:
    TensorRTInferenceEngine = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_first_json(text: str):
    """Parse the first complete JSON value ({...} or [...]) from a string."""
    text = text.strip()
    object_start = text.find("{")
    array_start = text.find("[")

    if object_start == -1 and array_start == -1:
        raise json.JSONDecodeError("No JSON value found", text, 0)

    if object_start == -1:
        start = array_start
    elif array_start == -1:
        start = object_start
    else:
        start = min(object_start, array_start)

    if start == -1:
        raise json.JSONDecodeError("No JSON value found", text, 0)

    opener = text[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escape = False
    quote = None

    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise json.JSONDecodeError("Unbalanced JSON delimiters", text, start)


SYSTEM_BIRA = """
You are BIRA, a friendly, enthusiastic, and helpful assistant designed to interpret object-grasping commands.
The user will give instructions and your task is to find which action to do based on the detected objects.
You must ALWAYS respond EXCLUSIVELY in valid JSON.

═══════════════════════════════════════════════════════════════
MODE DECISION TREE (APPLY IN ORDER):
═══════════════════════════════════════════════════════════════

1. STOP CHECK: Did the user ask to cancel, abort, or stop?
   → YES: Set mode='stop'. Respond: "Alright, I'm cancelling the operation."
   → NO: Continue to step 2.

2. INTELLIGIBILITY CHECK: Is the transcription empty or completely unintelligible?
   → YES: Set mode='repeat'. Ask user to repeat in a friendly manner.
   → NO: Continue to step 3.

3. ACTION CLARITY CHECK: Can you understand what ACTION the user wants?
    (e.g., pick, grab, bring, fetch, move, place, give/donne, take/prends, bring/apporte, etc.)
   → NO: Set mode='repeat'. Respond: "I'm sorry, I couldn't understand the action you want. Could you repeat?"
   → YES: Continue to step 4.

4. OBJECT MATCHING: How many candidate objects match the command from the provided candidate list?
   
   CASE A: 0 matches (no object in list matches the described object)
   → Set mode='clarification'. Respond with a friendly question asking the user to describe the object better
     or confirm if they see it. Examples:
       "I'm happy to help, but I don't see that object. Could you describe it differently?"
       "I'm not detecting the [OBJECT]. Is it visible in front of me?"
   
   CASE B: Exactly 1 match (one unique object matches the command)
   → Set mode='confirmation'. Provide the selected_candidate_index and respond with an enthusiastic confirmation.
     Always include the found object name in your response.
     Examples:
       "Alright, I'm picking up the [OBJECT]."
       "Got it, I'll grab the [OBJECT] for you."
   
   CASE C: Multiple matches (>1 candidate could match the command)
   → Set mode='clarification'. Ask for a disambiguating detail to narrow down which specific object.
     Suggest positional hints like: "the one on the left", "the top one", "the closest one", etc.
     Examples:
       "I found several [OBJECT]s. Could you tell me which one? (e.g., left, right, closest, farthest?)"
       "There are multiple [OBJECT]s. Which one would you like? (e.g., the one on the left, the one on top?)"

5. SCOPE CHECK: Is the user asking something outside object-grasping assistance?
   (e.g., "sing a song", "tell a joke", "dance", "what's the weather")
   → YES: Set request_scope='out_of_scope', mode='clarification', and explain briefly that you can only help with object-related commands.
   → NO: Set request_scope='in_scope'.

═══════════════════════════════════════════════════════════════
IMPORTANT CONSTRAINTS:
═══════════════════════════════════════════════════════════════
- If you select a candidate by index, ALWAYS return selected_candidate_index.
- selected_label and selected_label_id should match the candidate data when applicable.
- NEVER make assumptions about which object if multiple could match.
- NEVER invent object names that are not in the candidate list.
- For groups like "the stuff" or "the things", ask for clarification (mode='clarification').
- If candidate list is provided with positions, you may reference positions in clarification questions.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════
Always return STRICTLY VALID JSON, even if some fields are null or empty.
Required structure (single JSON object, not an array):
{
  "response": "...",
    "mode": "confirmation" or "clarification" or "repeat" or "stop",
    "request_scope": "in_scope" or "out_of_scope",
    "selected_candidate_index": integer or null,
    "selected_label": "<object_name_from_detected_list>" or null,
    "selected_label_id": integer or null
}
"""


class SLM_Manager:
    def __init__(
        self,
        model_name: str = "qwen3:1.7b",
        max_new_tokens: int = 500,
        temperature: float = 0.3,
        mode: str = "local",
        api_key: Optional[str] = None,
        prefer_tensorrt: Optional[bool] = None,
        tensorrt_engine_dir: Optional[str] = None,
        tensorrt_fallback_to_ollama: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mode = mode
        self.history = [{"role": "system", "content": SYSTEM_BIRA}]
        self.detections = None
        self.detection_candidates = None
        self.transcription = None
        self.prompt = None
        self.prefer_tensorrt = (
            _env_flag("USE_TENSORRT")
            if prefer_tensorrt is None
            else prefer_tensorrt
        )
        self.tensorrt_fallback_to_ollama = (
            _env_flag("TENSORRT_FALLBACK_TO_OLLAMA", default=True)
            if tensorrt_fallback_to_ollama is None
            else tensorrt_fallback_to_ollama
        )
        default_engine_dir = Path(__file__).resolve().parent / "tensorRT" / "tensorrt_models" / "engines"
        self.tensorrt_engine_dir = Path(tensorrt_engine_dir or default_engine_dir)
        self.trt_engine: Optional[Any] = None
        self.trt_ready = False
        self.model_loaded = False
        self.debug = _env_flag("SLM_DEBUG", default=False)

        if mode == "cloud":
            if not api_key:
                raise ValueError("API key must be provided for cloud mode.")

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

    def _init_tensorrt_engine(self) -> None:
        try:
            self.trt_engine = TensorRTInferenceEngine(engine_dir=str(self.tensorrt_engine_dir))
            self.trt_ready = bool(self.trt_engine.load_engine())

            if self.trt_ready:
                print(f"TensorRT ready (engine: {self.tensorrt_engine_dir})")
                return

            self.trt_engine = None
            if not self.tensorrt_fallback_to_ollama:
                raise RuntimeError(f"No TensorRT engine available in {self.tensorrt_engine_dir}")

            print(f"TensorRT engine not ready in {self.tensorrt_engine_dir}. Falling back to Ollama.")
        except Exception as exc:
            self.trt_ready = False
            self.trt_engine = None
            if not self.tensorrt_fallback_to_ollama:
                raise RuntimeError(f"TensorRT initialization failed: {exc}") from exc
            print(f"TensorRT unavailable ({exc}). Falling back to Ollama.")

    def _debug_log(self, message: str) -> None:
        if self.debug:
            print(f"[SLM_DEBUG] {message}")

    def _warmup_tensorrt_backend(self) -> None:
        if not self.trt_ready or not self.trt_engine:
            return

        print("Warming up TensorRT backend...")
        self.trt_engine.chat(
            messages=[{"role": "user", "content": "Reply with {}."}],
            max_new_tokens=8,
            temperature=0.0,
        )
        print("TensorRT backend ready.")

    def _warmup_ollama_backend(self) -> None:
        print(f"Starting model {self.model_name}...")
        self.client.generate(
            model=self.model_name,
            prompt="Reply with {}.",
            options={"num_predict": 8, "temperature": 0},
        )
        print("Model is awake.")

    def reset_conversation(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_BIRA}]
        self.detections = None
        self.detection_candidates = None
        self.transcription = None
        self.prompt = None

    def set_transcription(self, transcription: str) -> None:
        self.transcription = transcription

    def set_detections(
        self,
        detection_labels: Optional[list[int]] = None,
        detected_objects: Optional[list] = None,
        computer_vision=None,
    ) -> None:
        """
        Set the detected objects and resolve their labels to human-readable names.
        
        Args:
            detection_labels: List of YOLO label IDs
            detected_objects: List of detected objects with raw_label attribute
            computer_vision: ComputerVision instance for label name resolution
        """
        resolved: list[str] = []
        candidates: list[dict] = []

        if detection_labels and computer_vision:
            resolved.extend(
                computer_vision.get_label_name(label_id)
                for label_id in detection_labels
            )
        elif detection_labels:
            # Fallback for standalone usage (e.g., __main__) when ComputerVision is unavailable.
            resolved.extend(str(label_id) for label_id in detection_labels)

        if detected_objects and computer_vision:
            for index, obj in enumerate(detected_objects):
                raw = getattr(obj, "raw_label", None)
                if raw is None:
                    continue
                label_id = int(raw)
                label_name = computer_vision.get_label_name(label_id)
                resolved.append(label_name)

                position = getattr(obj, "position", None)
                if position is not None and len(position) >= 3:
                    pos_value = [float(position[0]), float(position[1]), float(position[2])]
                else:
                    pos_value = None

                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": label_id,
                        "position": pos_value,
                    }
                )

        if not candidates and detection_labels:
            for index, label_id in enumerate(detection_labels):
                label_name = str(label_id)
                if computer_vision:
                    label_name = computer_vision.get_label_name(label_id)
                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": int(label_id),
                        "position": None,
                    }
                )

        self.detections = list(dict.fromkeys(resolved)) if resolved else None
        self.detection_candidates = candidates if candidates else None

    def _build_prompt(self, transcription: str, detections: list[str], candidates: Optional[list[dict]]) -> str:
        candidates_text = json.dumps(candidates or [], ensure_ascii=False)
        return (
            "Analyze the following command and decide on the action to take: "
            f"'{transcription}'. The detected objects are: {detections}. "
            f"Candidate list (each candidate has index, label, label_id, and optional position [x, y, z]): {candidates_text}. "
            "If the user asks for something outside object-grasping assistance (e.g., singing, jokes, weather), "
            "set request_scope='out_of_scope' and keep mode='clarification'. "
            "When selecting one object, prefer returning selected_candidate_index. "
            "If you still cannot uniquely choose one object, ask a clarification question and keep mode='clarification'. "
            "Return ONLY one JSON object. Do not return markdown, explanation text, or arrays. "
            "Respond in JSON according to the rules."
        )

    def _parse_model_response(self, raw_response: str) -> dict:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = _parse_first_json(raw_response)

        if isinstance(parsed, list) and parsed:
            parsed = parsed[-1]

        if not isinstance(parsed, dict):
            raise ValueError("Model response is not a JSON object")

        mode = parsed.get("mode", "clarification")
        mode = str(mode).strip().lower()
        if mode not in {"confirmation", "clarification", "repeat", "stop"}:
            mode = "clarification"
        request_scope = parsed.get("request_scope", "in_scope")
        request_scope = str(request_scope).strip().lower()
        if request_scope not in {"in_scope", "out_of_scope"}:
            request_scope = "in_scope"
        text = parsed.get("response", "I didn't understand. Could you repeat?")
        selected_label = parsed.get("selected_label")
        if selected_label is not None:
            selected_label = str(selected_label).strip() or None

        selected_label_id = parsed.get("selected_label_id")
        if selected_label_id is not None:
            try:
                selected_label_id = int(selected_label_id)
            except (TypeError, ValueError):
                selected_label_id = None

        selected_candidate_index = parsed.get("selected_candidate_index")
        if selected_candidate_index is not None:
            try:
                selected_candidate_index = int(selected_candidate_index)
            except (TypeError, ValueError):
                selected_candidate_index = None

        return {
            "response": str(text),
            "mode": mode,
            "request_scope": request_scope,
            "selected_label": selected_label,
            "selected_label_id": selected_label_id,
            "selected_candidate_index": selected_candidate_index,
        }

    def run_inference(self) -> dict:
        transcription = self.transcription
        detections = self.detections
        candidates = self.detection_candidates

        if not transcription:
            return {
                "response": "I didn't hear a command. Could you repeat?",
                "mode": "repeat",
                "request_scope": "in_scope",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }

        if not detections:
            return {
                "response": "I don't see any relevant object. Could you clarify?",
                "mode": "clarification",
                "request_scope": "in_scope",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }

        self.prompt = self._build_prompt(transcription, detections, candidates)
        self._debug_log(
            "prompt_ready "
            f"chars={len(self.prompt)} "
            f"transcription={repr((transcription or '')[:120])} "
            f"detections_count={len(detections or [])} "
            f"candidates_count={len(candidates or [])}"
        )
        self._debug_log(f"prompt_preview={repr(self.prompt[:260])}")
        raw_response = self.generate_response(self.prompt)

        try:
            response = self._parse_model_response(raw_response)
        except (json.JSONDecodeError, ValueError) as err:
            print("SLM JSON parse error:", err)
            print("SLM raw response:", repr(raw_response))
            fallback_text = str(raw_response or "").strip()
            if fallback_text:
                response = {
                    "response": fallback_text,
                    "mode": "clarification",
                    "request_scope": "in_scope",
                    "selected_label": None,
                    "selected_label_id": None,
                    "selected_candidate_index": None,
                }
            else:
                response = {
                    "response": "I didn't understand. Could you repeat?",
                    "mode": "repeat",
                    "request_scope": "in_scope",
                    "selected_label": None,
                    "selected_label_id": None,
                    "selected_candidate_index": None,
                }

        # Guardrail: if user utterance is non-empty and objects are available,
        # avoid falling back to repeat loops unless input is truly unintelligible.
        if (
            response.get("mode") == "repeat"
            and transcription
            and str(transcription).strip()
            and detections
        ):
            response["mode"] = "clarification"
            if not str(response.get("response") or "").strip() or "repeat" in str(response.get("response", "")).lower():
                response["response"] = (
                    "I understood your request, but I need more details or I cannot find that exact object. "
                    "Could you specify which one you mean?"
                )

        if response["mode"] == "stop":
            self.reset_conversation()

        return response

    def load_model(self):
        if self.model_loaded:
            return

        tensorrt_ready = False
        if self.prefer_tensorrt:
            if self.trt_ready and self.trt_engine:
                self._warmup_tensorrt_backend()
                tensorrt_ready = True

                if not self.tensorrt_fallback_to_ollama:
                    print("TensorRT is ready. Skipping Ollama startup.")
                    self.model_loaded = True
                    return

            if not self.trt_ready and not self.tensorrt_fallback_to_ollama:
                raise RuntimeError(
                    "TensorRT was requested but no runnable engine is available. "
                    f"Expected assets in {self.tensorrt_engine_dir}."
                )

        if self.mode == "cloud":
            print(f"Checking model '{self.model_name}' through Ollama Cloud...")
            try:
                self._warmup_ollama_backend()
            except Exception as exc:
                if tensorrt_ready:
                    print(f"Ollama Cloud warmup skipped ({exc}). TensorRT remains available.")
                    self.model_loaded = True
                    return
                raise RuntimeError(
                    "Unable to contact Ollama Cloud. Check the API key and the model name."
                ) from exc

            print("Cloud model is reachable.")
            print("Python client ready.")
            self.model_loaded = True
            return

        print("Checking local Ollama server...")
        try:
            models_result = subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            if tensorrt_ready:
                print(f"Local Ollama warmup skipped ({exc}). TensorRT remains available.")
                self.model_loaded = True
                return
            raise RuntimeError("Ollama is not running. Start it with 'ollama serve'.") from exc

        models = models_result.stdout

        if self.model_name not in models:
            if tensorrt_ready:
                print(f"Model {self.model_name} not found in Ollama. TensorRT remains available.")
                self.model_loaded = True
                return
            raise RuntimeError(
                f"The model {self.model_name} does not exist in Ollama.\n"
                f"Create it with: ollama create {self.model_name} -f Modelfile"
            )

        try:
            self._warmup_ollama_backend()
        except Exception as exc:
            if tensorrt_ready:
                print(f"Local Ollama warmup skipped ({exc}). TensorRT remains available.")
                self.model_loaded = True
                return
            raise RuntimeError(
                f"Unable to warm up local model '{self.model_name}'."
            ) from exc

        print("Python client ready.")
        self.model_loaded = True

    def preload(self) -> None:
        self.load_model()

    @staticmethod
    def _extract_ollama_message(response) -> tuple[str, str, str]:
        """Return (content, thinking, done_reason) from dict-like or Pydantic Ollama responses."""
        data = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(data, dict):
            return "", "", ""

        message = data.get("message", {})
        if hasattr(message, "model_dump"):
            message = message.model_dump()

        if isinstance(message, dict):
            content = str(message.get("content") or "")
            thinking = str(message.get("thinking") or "")
        else:
            content = str(getattr(message, "content", "") or "")
            thinking = str(getattr(message, "thinking", "") or "")

        done_reason = str(data.get("done_reason") or "")
        return content, thinking, done_reason

    def generate_response(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})

        if self.prefer_tensorrt and self.trt_ready and self.trt_engine:
            try:
                response = self.trt_engine.chat(
                    messages=self.history,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                content = response.get("message", {}).get("content", "")
                if content:
                    return content

                if not self.tensorrt_fallback_to_ollama:
                    raise RuntimeError("TensorRT returned an empty response.")
            except Exception as exc:
                print(f"TensorRT failed ({exc}).")
                if not self.tensorrt_fallback_to_ollama:
                    raise

        response = self.client.chat(
            model=self.model_name,
            messages=self.history,
            format="json",
            options={
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        )
        content, thinking, done_reason = self._extract_ollama_message(response)
        self._debug_log(
            "chat_json "
            f"done_reason={done_reason or 'n/a'} "
            f"content_len={len(content)} "
            f"thinking_len={len(thinking)}"
        )
        if content.strip():
            return content

        # qwen3 can spend many tokens in `thinking` before writing assistant `content`.
        # If generation stopped due to token limit, retry once with a larger budget.
        if done_reason == "length":
            boosted_tokens = max(self.max_new_tokens * 4, 800)
            retry_long = self.client.chat(
                model=self.model_name,
                messages=self.history,
                format="json",
                options={
                    "temperature": self.temperature,
                    "num_predict": boosted_tokens,
                },
            )
            long_content, long_thinking, _ = self._extract_ollama_message(retry_long)
            _, _, long_done_reason = self._extract_ollama_message(retry_long)
            self._debug_log(
                "chat_json_retry_long "
                f"num_predict={boosted_tokens} "
                f"done_reason={long_done_reason or 'n/a'} "
                f"content_len={len(long_content)} "
                f"thinking_len={len(long_thinking)}"
            )
            if long_content.strip():
                return long_content
            if long_thinking.strip():
                return long_thinking

        # Some model/backends can return empty content when strict JSON mode is enforced.
        # Retry once without forced JSON output, then let the parser recover the first JSON value.
        print("SLM warning: empty chat response in JSON mode, retrying without forced JSON format.")
        retry = self.client.chat(
            model=self.model_name,
            messages=self.history,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        )
        retry_content, retry_thinking, retry_done_reason = self._extract_ollama_message(retry)
        self._debug_log(
            "chat_no_json_retry "
            f"done_reason={retry_done_reason or 'n/a'} "
            f"content_len={len(retry_content)} "
            f"thinking_len={len(retry_thinking)}"
        )
        if retry_content.strip():
            return retry_content
        if retry_thinking.strip():
            return retry_thinking

        # Last fallback using generate endpoint with the same prompt.
        fallback = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        )
        fallback_content = fallback.get("response", "")
        self._debug_log(
            "generate_fallback "
            f"response_len={len(str(fallback_content or ''))}"
        )
        return str(fallback_content or "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SLM_Manager inference locally")
    parser.add_argument("--model", dest="BIRA", default="qwen3:1.7b", help="Model name to use")
    parser.add_argument("--mode", default="local", choices=["local", "cloud"], help="Deployment mode")
    parser.add_argument("--api-key", default="", help="API key for cloud mode")
    parser.add_argument(
        "--prefer-tensorrt",
        action="store_true",
        help="Use the TensorRT backend when a runnable engine is configured.",
    )
    parser.add_argument(
        "--tensorrt-engine-dir",
        default=None,
        help="Path to the TensorRT engine directory.",
    )

    args = parser.parse_args()

    print(f"Initializing SLM_Manager (mode: {args.mode}, model: {args.BIRA})")
    manager = SLM_Manager(
        model_name=args.BIRA,
        mode=args.mode,
        api_key=args.api_key if args.api_key else None,
        prefer_tensorrt=args.prefer_tensorrt,
        tensorrt_engine_dir=args.tensorrt_engine_dir,
    )

    try:
        manager.load_model()
    except Exception as exc:
        print(f"Warning: model load failed: {exc}")
        print("Falling back to assuming the client is reachable...")

    print("\n--- Interactive SLM Test ---")
    print("Type 'q' or 'quit' to exit.")

    while True:
        try:
            transcription = input("\nEnter user command (transcription): ").strip()
            if transcription.lower() in ["q", "quit"]:
                break

            detections_input = input(
                "Enter detected objects (comma-separated labels, e.g. '0, 41, 39' for person, cup, bottle): "
            ).strip()
            if detections_input.lower() in ["q", "quit"]:
                break

            detection_labels = []
            if detections_input:
                try:
                    detection_labels = [int(value.strip()) for value in detections_input.split(",") if value.strip()]
                except ValueError:
                    print("Error: detected objects must be integers.")
                    continue

            manager.set_transcription(transcription)
            manager.set_detections(detection_labels=detection_labels)

            print("\n[Running inference...]")
            result = manager.run_inference()

            print("\n--- Result ---")
            print(f"Response: {result.get('response')}")
            print(f"Mode:     {result.get('mode')}")
            print("-------------")
        except KeyboardInterrupt:
            break

    print("\nExiting SLM test.")
