"""Prepare TensorRT assets and config for the BIRA SLM pipeline."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "tensorrt_models"
CONFIG_TEMPLATE_PATH = Path(__file__).resolve().parent / "tensorrt_config.json"


def detect_jetson_platform() -> Dict[str, Any]:
    """Return basic platform information and whether the host is a Jetson."""
    try:
        if os.path.exists("/etc/nv_tegra_release"):
            version_info = Path("/etc/nv_tegra_release").read_text(encoding="utf-8").strip()
            model = "Unknown Jetson"
            model_path = Path("/proc/device-tree/model")
            if model_path.exists():
                model = model_path.read_text(encoding="utf-8", errors="ignore").strip("\x00")
            return {
                "is_jetson": True,
                "model": model,
                "version_info": version_info,
                "architecture": platform.machine(),
            }
    except Exception as exc:
        logger.debug("Jetson detection failed: %s", exc)

    return {"is_jetson": False, "architecture": platform.machine()}


class TensorRTOptimizer:
    """Prepare a TensorRT workspace for the model used by `SLM_Manager`."""

    def __init__(
        self,
        model_name: str = "qwen3:1.7b",
        output_dir: Optional[str] = None,
        precision: str = "fp16",
        max_batch_size: int = 1,
        max_input_len: int = 512,
        max_output_len: int = 192,
        gpu_id: int = 0,
        auto_detect_jetson: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.gpu_id = gpu_id
        self.platform_info = detect_jetson_platform() if auto_detect_jetson else {"is_jetson": False}

        if self.platform_info.get("is_jetson") and "Orin Nano" in self.platform_info.get("model", ""):
            if self.max_input_len > 256:
                logger.warning("Reducing max_input_len from %s to 256 for Jetson Orin Nano.", self.max_input_len)
                self.max_input_len = 256
            if self.max_output_len > 128:
                logger.warning("Reducing max_output_len from %s to 128 for Jetson Orin Nano.", self.max_output_len)
                self.max_output_len = 128

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.engine_dir = self.output_dir / "engines"
        self.cache_dir = self.output_dir / "cache"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("TensorRT optimizer ready for %s", self.model_name)

    def check_dependencies(self) -> bool:
        """Check the Python dependencies needed to prepare a TensorRT workflow."""
        dependencies = {
            "tensorrt": "TensorRT",
            "torch": "PyTorch",
            "transformers": "Transformers",
        }

        missing = []
        for module_name, display_name in dependencies.items():
            try:
                __import__(module_name)
                logger.info("%s available", display_name)
            except ImportError:
                missing.append(display_name)
                logger.error("%s missing", display_name)

        if missing:
            logger.error("Missing dependencies: %s", ", ".join(missing))
            return False

        return True

    def export_ollama_model(self) -> bool:
        """Check that the source model exists in Ollama before conversion."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        except Exception as exc:
            logger.error("Unable to query Ollama: %s", exc)
            return False

        if self.model_name not in result.stdout:
            logger.error("Model %s was not found in Ollama.", self.model_name)
            logger.info("Download it first with: ollama pull %s", self.model_name)
            return False

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model %s is available for TensorRT conversion.", self.model_name)
        return True

    def _load_template_config(self) -> Dict[str, Any]:
        if not CONFIG_TEMPLATE_PATH.exists():
            return {}
        with CONFIG_TEMPLATE_PATH.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def convert_to_tensorrt(self) -> bool:
        """Create the TensorRT workspace and persist the merged config."""
        try:
            self.engine_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            config = self._load_template_config()
            config.setdefault("model_config", {})
            config["model_config"].update({"model_name": self.model_name})
            config.setdefault("optimization_config", {})
            config["optimization_config"].update(
                {
                    "precision": self.precision,
                    "max_batch_size": self.max_batch_size,
                    "max_input_length": self.max_input_len,
                    "max_output_length": self.max_output_len,
                    "use_fp16": self.precision == "fp16",
                    "use_int8": self.precision == "int8",
                    "use_int4": self.precision == "int4",
                    "jetson_optimized": bool(self.platform_info.get("is_jetson")),
                }
            )
            config.setdefault("hardware_config", {})
            config["hardware_config"].update(
                {
                    "gpu_id": self.gpu_id,
                    "platform": "jetson_orin_nano" if self.platform_info.get("is_jetson") else platform.system().lower(),
                }
            )
            config.setdefault("paths", {})
            config["paths"].update(
                {
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "engine_dir": str(self.engine_dir),
                    "cache_dir": str(self.cache_dir),
                }
            )
            config.setdefault("runtime", {})
            if not config["runtime"].get("runner_command"):
                config["runtime"]["runner_command"] = os.getenv("TENSORRT_RUNNER_COMMAND")

            config_path = self.engine_dir / "config.json"
            with config_path.open("w", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2)

            logger.info("TensorRT config written to %s", config_path)
            logger.info("The actual engine build still needs to be done by your TensorRT toolchain.")
            return True
        except Exception as exc:
            logger.error("TensorRT conversion setup failed: %s", exc)
            return False

    def optimize(self) -> bool:
        """Run the preparation pipeline used before enabling TensorRT in `SLM_Manager`."""
        logger.info("Starting TensorRT optimization workflow...")
        if not self.check_dependencies():
            return False
        if not self.export_ollama_model():
            return False
        if not self.convert_to_tensorrt():
            return False
        logger.info("TensorRT workspace prepared in %s", self.output_dir)
        return True

    def get_engine_info(self) -> Dict[str, Any]:
        """Read the generated config from the engine directory."""
        config_path = self.engine_dir / "config.json"
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimizer = TensorRTOptimizer()
    if optimizer.optimize():
        print(json.dumps(optimizer.get_engine_info(), indent=2))
    else:
        raise SystemExit(1)
