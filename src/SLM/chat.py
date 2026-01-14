import os
from dotenv import load_dotenv
from ollama import chat 
from configure_test_ollama import (
    check_ollama_installed,
    pull_source_model_if_needed,
    build_model_from_modelfile,
    stop_model,
)

def get_temperature(city: str) -> str:
  """Get the current temperature for a city
  
  Args:
    city: The name of the city

  Returns:
    The current temperature for the city
  """
  temperatures = {
    'New York': '22°C',
    'London': '15°C',
  }
  return temperatures.get(city, 'Unknown')

class SLM_Manager:
    def __init__(self):
        load_dotenv()
        self.source_model_name = os.getenv("SOURCE_MODEL_NAME", "llama3.2:1b")
        self.target_model_name = os.getenv("TARGET_MODEL_NAME", "BIRA")
        self.messages = []

        self._ensure_ollama_ready()

    def _ensure_ollama_ready(self) -> None:
        """Ensure Ollama CLI and the target model are available before use."""
        try:
            check_ollama_installed()
            needs_build = pull_source_model_if_needed(self.target_model_name, self.source_model_name)
            new_model_file = os.getenv("NEW_MODEL_FILE", "True").lower() == "true"
            if needs_build or new_model_file:
                build_model_from_modelfile(self.target_model_name, self.source_model_name)
        except SystemExit as exc:
            raise RuntimeError("Échec de la configuration d'Ollama via configure_test_ollama.") from exc
    
    def stop_model(self) -> None:
        """Stop the running target model."""
        stop_model(self.target_model_name)

    def stream_chat(self, new_message: str) -> None:
        """Stream chat responses from the target model."""
        self.messages.append({
            'role': 'user',
            'content': new_message,
        })
        
        while True:
            stream = chat(
                model='BIRA',
                messages=self.messages,
                tools=[get_temperature],
                stream=True,
                think=True,
            )

            thinking = ''
            content = ''
            tool_calls = []

            done_thinking = False
            # accumulate the partial fields
            for chunk in stream:
                if chunk.message.thinking:
                    thinking += chunk.message.thinking
                    print(chunk.message.thinking, end='', flush=True)
                if chunk.message.content:
                    if not done_thinking:
                        done_thinking = True
                        print('\n')
                    content += chunk.message.content
                print(chunk.message.content, end='', flush=True)
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)
                    print(chunk.message.tool_calls)

            # append accumulated fields to the messages
            if thinking or content or tool_calls:
                self.messages.append({'role': 'assistant', 'thinking': thinking, 'content': content, 'tool_calls': tool_calls})

            if not tool_calls:
                break

            for call in tool_calls:
                if call.function.name == 'get_temperature':
                    result = get_temperature(**call.function.arguments)
                else:
                    result = 'Unknown tool'
                self.messages.append({'role': 'tool', 'tool_name': call.function.name, 'content': result})
        print("\n--- Conversation ended ---\n")
        
manager = SLM_Manager()

while True:
    try:
        prompt = input("Votre prompt: ")
        
        # Get response and add to conversation history
        manager.stream_chat(prompt)
        
    except KeyboardInterrupt:
        manager.stop_model()
        print("\nExiting...")
        break
