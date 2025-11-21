import os
from ollama import Client
from ollama import ChatResponse

OLLAMA_API_KEY = ''

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY}
)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

response: ChatResponse = client.chat('gpt-oss:120b', messages=messages)
print(response['message']['content'])
