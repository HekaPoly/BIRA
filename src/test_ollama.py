from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)
response = client.chat(model='BIRA', messages=[
  {
    'role': 'user',
    'content': 'Can you please give me the apple in front of you?',
  },
])

print(response)