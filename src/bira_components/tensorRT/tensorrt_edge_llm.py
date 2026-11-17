from typing import Any, Dict
from tensorrt_edgellm.server import LLM, SamplingParams
from tensorrt_edgellm.server.runtime.engine_build import BuildOptions
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
                # under experimental/server/config.py
                "method": "",
                "num_speculative_tokens": 0,
                "model": "",
                "draft_top_k": 0,
                "draft_model": "",
                "verify_tokens": 0
            }
        
        build_options = BuildOptions(
            # TODO: Build configuration goes here
            # under experimental/server/runtime/engine_build.py
            spec_type = "none",
            draft_model_dir = "",
            max_input_len = None,
            max_kv_cache_capacity = None,
            max_batch_size = None,
            max_spec_verify_tokens = None,
            max_spec_draft_tokens = None,
            max_image_tokens = None,
            max_image_tokens_per_image = None,
            tp_size = None,
            plugin_path = ""
        )

        # This automatically builds the engine if necessary.
        self.llm = LLM(
            model=self.model,
            cache_dir=output_dir or self.cache_dir,
            # max_input_len=self.max_input_len,
            # max_batch_size=self.max_batch_size,
            # max_kv_cache_capacity=self.max_kv_cache_capacity,
            on_progress=print if debug else None
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
            tool_choice=tool_choice
            # sampling_params=sampling_params
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
    trt = TensorRTModel(model="qwen3:1.7b")
    trt.build_engine(output_dir="trt_cache", debug=True)
    trt.launch_server()
    messages = []
    print(f"Server is running. You can now send messages to {trt.model}.")
    
    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        output = trt.chat(messages=messages)
        # print(output.finish_reason)
        print(f"reasoning: {output.reasoning}")
        print(f"answer: {output.text}")
        messages.append({"role": "assistant", "content": output.text})
