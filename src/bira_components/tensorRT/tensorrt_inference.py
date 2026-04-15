"""TensorRT inference wrapper compatible with the current `SLM_Manager`."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_DIR = Path(__file__).resolve().parent / "tensorrt_models" / "engines"
DEFAULT_CONFIG_TEMPLATE = Path(__file__).resolve().parent / "tensorrt_config.json"


class TensorRTInferenceEngine:
    """Generic TensorRT chat wrapper driven by a local runner command."""

    def __init__(self, engine_dir: Optional[str] = None, device_id: int = 0):
        self.engine_dir = Path(engine_dir) if engine_dir else DEFAULT_ENGINE_DIR
        self.device_id = device_id
        self.config: Dict[str, Any] = {}
        self.is_loaded = False
        self.runner_command: List[str] = []

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.engine_dir / "config.json"
        if not config_path.exists():
            if DEFAULT_CONFIG_TEMPLATE.exists():
                logger.info("TensorRT config not found at %s; bootstrapping from %s", config_path, DEFAULT_CONFIG_TEMPLATE)
                with DEFAULT_CONFIG_TEMPLATE.open("r", encoding="utf-8") as stream:
                    return json.load(stream)

            logger.warning("TensorRT config not found at %s", config_path)
            return {}

        with config_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _write_config(self, config: Dict[str, Any]) -> None:
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.engine_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)

    def _bootstrap_engine_dir(self) -> None:
        if self.engine_dir.exists():
            return

        logger.info("TensorRT engine directory not found at %s; creating it from the bundled template.", self.engine_dir)
        config = self._load_config()
        if not config:
            self.engine_dir.mkdir(parents=True, exist_ok=True)
            return

        runtime_cfg = config.setdefault("runtime", {})
        env_runner = os.getenv("TENSORRT_RUNNER_COMMAND")
        if env_runner:
            runtime_cfg["runner_command"] = env_runner

        self._write_config(config)

    def _resolve_runner_command(self) -> List[str]:
        runtime_cfg = self.config.get("runtime", {})
        configured = os.getenv("TENSORRT_RUNNER_COMMAND") or runtime_cfg.get("runner_command")

        if isinstance(configured, list):
            return [str(part) for part in configured]
        if isinstance(configured, str) and configured.strip():
            return shlex.split(configured)

        for candidate in ("trtllm-serve", "trtllm", "tensorrt_llm"):
            resolved = shutil.which(candidate)
            if resolved:
                logger.info("TensorRT runner auto-detected: %s", resolved)
                return [resolved]

        return []

    def load_engine(self) -> bool:
        """Validate the TensorRT assets and the runner command."""
        self._bootstrap_engine_dir()

        self.config = self._load_config()
        if not self.config:
            logger.warning("TensorRT configuration could not be loaded for %s", self.engine_dir)
            return False

        if not self.engine_dir.exists():
            logger.warning("TensorRT engine directory not found: %s", self.engine_dir)
            return False

        self.runner_command = self._resolve_runner_command()
        if not self.runner_command:
            config_path = self.engine_dir / "config.json"
            logger.warning(
                "TensorRT config is present at %s, but runtime.runner_command is not set. "
                "Set a real local runner command in that file or export TENSORRT_RUNNER_COMMAND.",
                config_path,
            )
            return False

        self.is_loaded = True
        logger.info("TensorRT engine is ready via runner: %s", " ".join(self.runner_command))
        return True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 192,
        temperature: float = 0.4,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """Generate text by delegating to the configured TensorRT runner."""
        if not self.is_loaded and not self.load_engine():
            raise RuntimeError("TensorRT engine is not ready.")

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "engine_dir": str(self.engine_dir),
            "device_id": self.device_id,
        }

        start_time = time.time()
        result = subprocess.run(
            self.runner_command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            stderr = result.stderr.strip() or "unknown TensorRT runner error"
            raise RuntimeError(stderr)

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("TensorRT runner returned no output.")

        generated_text = self._extract_generated_text(stdout)
        return {
            "generated_text": generated_text,
            "prompt": prompt,
            "inference_time": elapsed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "engine": "TensorRT",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 192,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """Expose an Ollama-like chat interface for `SLM_Manager`."""
        prompt = self._format_chat_prompt(messages)
        result = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {
            "message": {
                "role": "assistant",
                "content": result["generated_text"],
            },
            "total_duration": int(result["inference_time"] * 1e9),
            "load_duration": 0,
            "eval_count": max_new_tokens,
            "engine": "TensorRT",
        }

    def _extract_generated_text(self, stdout: str) -> str:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout

        if isinstance(parsed, dict):
            if "generated_text" in parsed:
                return str(parsed["generated_text"])
            if "response" in parsed:
                return str(parsed["response"])
            if "message" in parsed and isinstance(parsed["message"], dict):
                return str(parsed["message"].get("content", ""))

        raise RuntimeError("TensorRT runner output must include generated_text, response, or message.content.")

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_loaded": self.is_loaded,
            "config": self.config,
            "device_id": self.device_id,
            "engine_dir": str(self.engine_dir),
            "runner_command": self.runner_command,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = TensorRTInferenceEngine()
    if not engine.load_engine():
        raise SystemExit("TensorRT engine is not ready.")

    response = engine.chat(
        messages=[{"role": "user", "content": "Can you give me the banana in front of you?"}],
        temperature=0.3,
    )
    print(response["message"]["content"])
