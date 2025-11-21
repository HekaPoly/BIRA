import os
from ollama import Client
from ollama import ChatResponse

OLLAMA_API_KEY = '747aadbe08f24aa5b2898948925dd80a.0hw2fdvQjib4R9VNLbxZVzHw'

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
