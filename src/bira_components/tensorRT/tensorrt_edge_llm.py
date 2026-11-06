from experimental.server import LLM, ContextCacheConfig, SamplingParams, BuildOptions
from pathlib import Path
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "trt_cache"

class TensorRTModel:
    def __init__(
        self,
        model: str,
        max_input_len: int = 4096,
        max_kv_cache_capacity: int = 8192,
        max_batch_size: int = 1,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        self.model = model
        self.max_input_len = max_input_len
        self.max_kv_cache_capacity = max_kv_cache_capacity
        self.max_batch_size = max_batch_size
        self.cache_dir = cache_dir
        self.llm = None

    def build_engine(
        self,
        output_dir: str = None,
        spec_type: str = None,
        draft_model_dir: str = None,
        debug: bool = False,
    ) -> None:
        speculative_config = None

        if spec_type is not None:
            speculative_config = {
                # TODO: Speculative decoding
            }

            if draft_model_dir is not None:
                speculative_config["model"] = draft_model_dir
        
        build_options = BuildOptions(
            # TODO: Build configuration goes here
        )

        # This automatically builds the engine if necessary.
        self.llm = LLM(
            model=self.model,
            cache_dir=output_dir or self.cache_dir,
            build_options=build_options,
            max_input_len=self.max_input_len,
            max_batch_size=self.max_batch_size,
            max_kv_cache_capacity=self.max_kv_cache_capacity,
            on_progress=print if debug else None,
        )

    def launch_server(
        self,
        port: int = 8000,
        host: str = "0.0.0.0"
    ) -> None:

        if self.llm is None:
            raise RuntimeError(
                "Model has not been initialized. Call build_engine() first."
            )

        self.llm.serve(
            host=host,
            port=port,
            enable_auto_tool_choice=True
        )
    
    def chat(
        self,
        messages: list,
        tool_choice: str | Dict[str, Any] | None = None
    ) -> list:
        sampling_params = SamplingParams(
            # TODO: set sampling parameters based on user input or defaults
        )
        if self.llm is None:
            raise RuntimeError(
                "Model has not been initialized. Call build_engine() first."
            )

        return self.llm.chat(
            messages=messages,
            tool_choice=tool_choice,
            sampling_params=sampling_params
        )

    def close(self):
        if self._closed:
            return

        self._closed = True

        if self.llm is not None:
            self.llm.close()
            self.llm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

if __name__ == "__main__":
    print("hello world")
