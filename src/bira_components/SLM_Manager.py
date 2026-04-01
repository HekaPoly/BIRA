from __future__ import annotations

from pathlib import Path
from typing import Optional
import argparse
import json
import os
import subprocess

from cv_viewer import labels
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


def _parse_first_json(text: str) -> dict:
    """Parse the first complete JSON object from a string."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

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
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise json.JSONDecodeError("Unbalanced braces", text, start)


SYSTEM_BIRA = """
You are BIRA, a friendly, enthusiastic, and helpful assistant designed to interpret object-grasping commands.
The user will give instructions and your task is to find which action to do based on the detected objects.
If the object is not detected, you must ask for clarification.
You must ALWAYS respond EXCLUSIVELY in valid JSON.

FUNDAMENTAL RULES:

Response:
   - Provide a confirmation sentence in the first person, with a friendly and enthusiastic tone.
   - Rephrase the action in an active way, including the identified object.
   - Examples:
       "Alright, I'm [ACTION] the [OBJECT]."
       "Got it, I'll [ACTION] the [OBJECT] for you."

   - If the command is too vague (imprecise object, unclear term, unclear action), ask for clarification enthusiastically.
   - If the request refers to a group of objects (e.g., "the stuff", "the things"), ask for clarification.
   - If the mentioned objects cannot be detected or do not seem to be present, ask for clarification.
   - If the user wants to eat a specific [OBJECT], you must confirm by responding that you will bring the food [OBJECT].
   - Examples:
       "I'm happy to help, but I do not see [OBJECT]. Could you specify which one you mean?"
       "I'm happy to help, but I can't seem to identify the [OBJECT]. Could you describe it a bit more?"
       "I'm happy to help, but I can't seem to identify the object. Could you describe it a bit more?"

OUTPUT FORMAT:
Always return strictly valid JSON, even if some fields are null or empty.

IT IS IMPERATIVE TO STRICTLY FOLLOW THE STRUCTURE BELOW AND MAKE SURE THE MODE IS PRESENT.
Structure:
[{
  "response": "...",
  "mode": "confirmation" | "clarification" | "stop"
},]
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
        self.trt_engine: Optional[TensorRTInferenceEngine] = None
        self.trt_ready = False
        self.model_loaded = False

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
        self.transcription = None
        self.prompt = None

    def set_transcription(self, transcription: str) -> None:
        self.transcription = transcription

    def set_detections(
        self,
        detection_labels: Optional[list[int]] = None,
        detected_objects: Optional[list] = None,
    ) -> None:
        resolved: list[str] = []

        if detection_labels:
            resolved.extend(labels.labelDict[label_id] for label_id in detection_labels if label_id in labels.labelDict)

        if detected_objects:
            for obj in detected_objects:
                raw = getattr(obj, "raw_label", None)
                if raw is None:
                    continue
                key = int(raw)
                if key in labels.labelDict:
                    resolved.append(labels.labelDict[key])

        self.detections = list(dict.fromkeys(resolved)) if resolved else None

    def _build_prompt(self, transcription: str, detections: list[str]) -> str:
        return (
            "Analyze the following command and decide on the action to take: "
            f"'{transcription}'. The detected objects are: {detections}. "
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
        text = parsed.get("response", "I didn't understand. Could you repeat?")
        return {"response": str(text), "mode": str(mode)}

    def run_inference(self) -> dict:
        transcription = self.transcription
        detections = self.detections

        if not transcription:
            return {"response": "I didn't hear a command. Could you repeat?", "mode": "clarification"}

        if not detections:
            return {"response": "I don't see any relevant object. Could you clarify?", "mode": "clarification"}

        self.prompt = self._build_prompt(transcription, detections)
        raw_response = self.generate_response(self.prompt)

        try:
            response = self._parse_model_response(raw_response)
        except (json.JSONDecodeError, ValueError) as err:
            print("SLM JSON parse error:", err)
            response = {"response": "I didn't understand. Could you repeat?", "mode": "clarification"}

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
            options={
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "format": "json",
            },
        )
        return response["message"]["content"]


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
