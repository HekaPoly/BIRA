from ollama import Client
import subprocess
import ollama

# subprocess.run(["pip", "install", "ollama"], check=True)

# Check if any models are installed
list_result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
models = list_result.stdout.splitlines()[1:]
# Check if BIRA:latest model is available
bira_available = any("BIRA:latest" in model for model in models)
print(bira_available)

# if "llama3.2:latest" not in list_result.stdout and "BIRA:latest" not in list_result.stdout:
#   # No models installed, pull the model
#   subprocess.run(["ollama", "pull", model_name], check=True)

model_name = "BIRA:latest"

# Start ollama run in a separate process and wait for completion
process = subprocess.run([f"ollama", "run", model_name], capture_output=True, text=True)

client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)
response = client.chat(model='BIRA', messages=[
  {
    'role': 'user',
    'content': 'Can you please give me the banana behind you?',
  },
])

print(response)