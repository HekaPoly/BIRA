from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
class SLM_Manager:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the SLM Manager with a Llama 3.2 model.

        Args:
            model_name (str): Hugging Face model ID or local path.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print("Model loaded successfully.")

    def generate_response(self, prompt, max_new_tokens=256, temperature=0.7):
        """
        Generate a response from the model.

        Args:
            prompt (str): Input prompt.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: Generated response.
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def chat(self):
        """Interactive chat loop."""
        print("Starting chat with Llama 3.2. Type 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                break
            response = self.generate_response(user_input)
            print(f"Llama 3.2: {response}")

if __name__ == "__main__":
    slm = SLM_Manager()
    slm.load_model()
    slm.chat()
