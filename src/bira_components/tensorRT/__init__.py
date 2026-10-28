"""TensorRT helpers for the BIRA SLM stack."""

from .tensorrt_inference import TensorRTInferenceEngine
from .tensorrt_optimizer import TensorRTOptimizer

__all__ = ["TensorRTOptimizer", "TensorRTInferenceEngine"]
__version__ = "1.0.0"
