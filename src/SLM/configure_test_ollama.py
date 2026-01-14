from ollama import Client
import subprocess
import os
from dotenv import load_dotenv

def check_ollama_installed():
    try:
        version_result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        print(f"✅ Ollama is installed: {version_result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Ollama is not installed or not found in PATH")
        try:
            print("Installing ollama via pip...")
            subprocess.run(["pip", "install", "ollama"], check=True)
            print('✅ Ollama installed successfully.')
        except subprocess.CalledProcessError:
            print("❌ Failed to install ollama")
            exit(1) 

def pull_source_model_if_needed(target_model_name="BIRA", source_model_name="llama3.2:1b"):
    list_result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    models = list_result.stdout.splitlines()
  
    # Check if source model is available
    if not any(source_model_name in model for model in models):
        # No models installed, pull the model
        print(f"⚠️ {source_model_name} not available.")
        print(f"Pulling model {source_model_name}...")
        try:
            subprocess.run(["ollama", "pull", source_model_name], check=True)
            print(f"✅ Model {source_model_name} pulled successfully.")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to pull model {source_model_name}")
            exit(1)
    else:
        print(f"✅ Model {source_model_name} is available.")
  
  # Check if target model is available
    if not any(target_model_name in model for model in models):
        print(f"⚠️ {target_model_name} not available. It needs to be built from Modelfile.")
        return True
    else:
        print(f"✅ Model {target_model_name} is available. No need to build.")
        return False


def build_model_from_modelfile(target_model_name="BIRA", source_model_name="llama3.2:1b"):
    # Resolve Modelfile relative to this script so it works from any cwd
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelfile_path = os.path.join(base_dir, "Modelfile")
    if os.path.exists(modelfile_path):
        with open(modelfile_path, 'r') as file:
            content = file.read()
            if f"FROM {source_model_name}" in content:
                print(f"✅ Modelfile contains 'FROM {source_model_name}'")
                print("🛠️ Creating model from Modelfile...")
                try:
                    subprocess.run(["ollama", "create", target_model_name, "-f", modelfile_path], check=True)
                    print(f"✅ Model {target_model_name} built successfully from Modelfile.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to build model {target_model_name} from Modelfile")
                    print(f"Error details: {e}")
                    exit(1)
            else:
                print(f"❌ Modelfile does not contain 'FROM {source_model_name}'. Please update it.")
                exit(1)
    else:
        print("❌ Modelfile not found")
        exit(1)

def test_ollama_run(target_model_name="BIRA", ollama_api_url="http://localhost:11434", test_response_time=True):
    # Start ollama run in a separate process and wait for completion
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
        return False
  
    if test_response_time:
        print("⏱️ Testing response time for prompts...\n")
        test_prompts = ["Peux-tu me passer la banane derrière toi, s'il te plaît?", "S'il te plaît?", "Merci!"]
    
        client = Client(
            host=ollama_api_url,
            headers={'x-some-header': 'some-value'}
        )

        for idx, prompt in enumerate(test_prompts):
            print(f"📝 Sending test prompt #{idx+1} to model: {prompt}")
            response = client.chat(model=target_model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ], options={"temperature": 1})

            print(f"🤖{target_model_name}: {response.message.content}\n")
            print(f"Total Duration: {response.total_duration / 1e9:.2f}s")
            print(f"Load Duration: {response.load_duration / 1e9:.2f}s")
            print(f"Number of tokens evaluated in inference: {response.eval_count}")
            print(f"Number of tokens evaluated in the prompt: {response.prompt_eval_count}\n")

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

    source_model_name = os.getenv("SOURCE_MODEL_NAME", "llama3.2")
    target_model_name = os.getenv("TARGET_MODEL_NAME", "BIRA")
    new_model_file = os.getenv("NEW_MODEL_FILE", "True").lower() == "true"
    test_response_time = os.getenv("TEST_RESPONSE_TIME", "True").lower() == "true"
    ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    
    check_ollama_installed()
    
    # Check if target model is available
    if new_model_file or pull_source_model_if_needed(target_model_name, source_model_name):
        build_model_from_modelfile(target_model_name, source_model_name)
        
    test_ollama_run(target_model_name, ollama_api_url, test_response_time)
    
