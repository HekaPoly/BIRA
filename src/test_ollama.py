from ollama import Client
import subprocess
import os

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