from ollama import Client
import subprocess
import os
from dotenv import load_dotenv


def run_model(target_model_name="BIRA"):
    try:
        # Check if model is already running
        ps_result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        running_models = ps_result.stdout.splitlines()

        # Check if target model is already running
        if not any(target_model_name in model for model in running_models):
            print(f"🦿 Starting ollama run for model: {target_model_name}...")
            subprocess.run(
                [f"ollama", "run", target_model_name],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            print(f"✅ Successfully started ollama run for model: {target_model_name}")
        else:
            print(f"🧐 Model {target_model_name} is already running. Stop it before starting a new instance by running 'ollama stop {target_model_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run model {target_model_name}")
        print(f"Error details: {e}")
        exit(1)


def create_client(ollama_api_url="http://localhost:11434"):
    client = Client(
        host=ollama_api_url,
        headers={'x-some-header': 'some-value'}
    )
    return client


def prompt_model(client, target_model_name, messages):
    print("Waiting for response...\n")
    response = client.chat(model=target_model_name, messages=messages, options={"temperature": 1})
    print(f"🤖{target_model_name}: {response.message.content}\n")
    print(f"Total Duration: {response.total_duration / 1e9:.2f}s")
    print(f"Load Duration: {response.load_duration / 1e9:.2f}s")
    return response.message.content


def stop_model(target_model_name="BIRA"):
    try:
        print(f"🛑 Stopping model: {target_model_name}...")
        subprocess.run(
            ["ollama", "stop", target_model_name],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print(f"✅ Successfully stopped model: {target_model_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to stop model {target_model_name}. Please stop it manually before starting a new instance by running 'ollama stop {target_model_name}'.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    load_dotenv()
    target_model_name = os.getenv("TARGET_MODEL_NAME", "BIRA")
    ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

    run_model(target_model_name)

    client = create_client(ollama_api_url)
    
    # Initialize conversation history
    messages = []

    while True:
        try:
            prompt = input("Votre prompt: ")
            
            # Add user message to conversation history
            messages.append({
                'role': 'user',
                'content': prompt,
            })
            
            # Get response and add to conversation history
            response_content = prompt_model(client, target_model_name, messages)
            messages.append({
                'role': 'assistant',
                'content': response_content,
            })
            
        except KeyboardInterrupt:
            stop_model(target_model_name)
            print("\nExiting...")
            break
