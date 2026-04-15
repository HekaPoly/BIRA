from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from bira_components.tensorRT import TensorRTInferenceEngine
except Exception:
    TensorRTInferenceEngine = None


class TensorRT_Manager:
    """Encapsulates TensorRT availability, lifecycle, and chat calls."""

    def __init__(self, mode: str, prefer_tensorrt: bool, engine_dir: Optional[str] = None):
        self.mode = mode
        self.prefer_tensorrt = prefer_tensorrt
        default_engine_dir = Path(__file__).resolve().parent / "tensorRT" / "tensorrt_models" / "engines"
        self.engine_dir = Path(engine_dir or default_engine_dir)
        self.engine = None
        self.ready = False

    def initialize(self) -> None:
        if self.mode != "local" and self.prefer_tensorrt:
            print("TensorRT is only available in local mode. Falling back to Ollama.")
            self.prefer_tensorrt = False
            return

        if not self.prefer_tensorrt:
            return

        if TensorRTInferenceEngine is None:
            print("TensorRT support is unavailable. Falling back to Ollama.")
            self.prefer_tensorrt = False
            return

        try:
            self.engine = TensorRTInferenceEngine(engine_dir=str(self.engine_dir))
            self.ready = bool(self.engine.load_engine())
            if self.ready:
                print(f"TensorRT ready (engine: {self.engine_dir})")
            else:
                self.engine = None
                print(
                    f"TensorRT engine not ready in {self.engine_dir}. "
                    "The config is missing runtime.runner_command, so Ollama will be used instead."
                )
        except Exception as exc:
            self.ready = False
            self.engine = None
            print(f"TensorRT unavailable ({exc}). Falling back to Ollama.")

    def chat(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        if not (self.prefer_tensorrt and self.ready and self.engine):
            return ""

        response = self.engine.chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        message = response.get("message", {}) if isinstance(response, dict) else {}
        return str(message.get("content") or "")

    def warmup(self) -> None:
        if self.prefer_tensorrt and self.ready and self.engine:
            self.engine.chat(
                messages=[{"role": "user", "content": "Reply with {}."}],
                max_new_tokens=8,
                temperature=0.0,
            )
