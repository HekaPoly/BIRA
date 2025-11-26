from ollama import Client
import subprocess
import os
from dotenv import load_dotenv

def check_ollama_installed():
  try:
    version_result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True)
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


def pull_source_model_if_needed(target_model_name="BIRA", source_model_name="llama3.2"):
  list_result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
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


def build_model_from_modelfile(target_model_name="BIRA", source_model_name="llama3.2"):
  modelfile_path = "./SLM/Modelfile"
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


def check_model_source(target_model_name="BIRA", expected_source="llama3.2"):
    """Check if the model is based on the expected source model"""
    try:
        show_result = subprocess.run(["ollama", "show", target_model_name], capture_output=True, text=True, check=True)
        output = show_result.stdout
        
        # Extract version from expected_source (e.g., "llama3.2" -> "3.2")
        if expected_source.lower().startswith("llama"):
            expected_version = expected_source.lower().replace("llama", "")
        else:
            expected_version = expected_source
        
        # Look for architecture and license information
        architecture_found = False
        version_found = False
        
        for line in output.splitlines():
            original_line = line.strip()
            line_lower = line.strip().lower()
            
            # Check architecture line - should be exactly "llama"
            if "architecture" in line_lower and "llama" in line_lower:
                architecture_found = True
                print(f"✅ Architecture: {original_line}")
            
            # Check license for specific LLAMA version
            if "llama" in line_lower and "license" in line_lower and expected_version in line_lower:
                version_found = True
                print(f"✅ License confirms LLAMA {expected_version}: {original_line}")
        
        if architecture_found and version_found:
            print(f"✅ {target_model_name} is confirmed to be based on LLAMA {expected_version}")
            return True
        elif architecture_found:
            print(f"⚠️ {target_model_name} is based on llama but version doesn't match {expected_version}")
            # Show what version it actually is
            for line in output.splitlines():
                if "llama" in line.lower() and "license" in line.lower():
                  print(f"   Found: {line.strip()}")
            return False
        else:
            print(f"❌ {target_model_name} is not based on llama architecture")
            print(f"Model info:\n{output}")
            return False
        
    except subprocess.CalledProcessError:
        print(f"❌ Model {target_model_name} not found")
        return False
    except FileNotFoundError:
        print("❌ Ollama command not found")
        return False

def test_ollama_run(target_model_name="BIRA"):
  # Start ollama run in a separate process and wait for completion
  try:
    # Check if model is already running
    ps_result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    running_models = ps_result.stdout.splitlines()
    
    # Check if target model is already running
    if not any(target_model_name in model for model in running_models):
      print(f"🦿 Starting ollama run for model: {target_model_name}...")
      subprocess.run([f"ollama", "run", target_model_name], capture_output=True, text=True, check=True)
      print(f"✅ Successfully started ollama run for model: {target_model_name}")
    else:
      print(f"🧐 Model {target_model_name} is already running. Stop it before starting a new instance by running 'ollama stop {target_model_name}'.")

  except subprocess.CalledProcessError as e:
    print(f"❌ Failed to run model {target_model_name}")
    print(f"Error details: {e}")
    return False
  
  test_prompts = ["Peux-tu me passer la banane derrière toi, s'il te plaît?", "S'il te plaît?", "Merci!"]
  
  client = Client(
    host='http://localhost:11434',
    headers={'x-some-header': 'some-value'}
  )

  for idx, prompt in enumerate(test_prompts):
    print(f"📝 Sending test prompt #{idx+1} to model: {prompt}")
    response = client.chat(model=target_model_name, messages=[
      {
        'role': 'user',
        'content': prompt,
      },
  ])

    print(f"📝 Response #{idx+1} from model:")
    print(f"Model: {response.model}")
    print(f"Content: {response.message.content}")
    print(f"Role: {response.message.role}")
    print(f"Total Duration: {response.total_duration / 1e9:.2f}s")
    print(f"Load Duration: {response.load_duration / 1e9:.2f}s")
    print(f"Eval Count: {response.eval_count}")
    print(f"Prompt Eval Count: {response.prompt_eval_count}")


def main():
  load_dotenv()

  source_model_name = os.getenv("SOURCE_MODEL_NAME", "llama3.2")
  target_model_name = os.getenv("TARGET_MODEL_NAME", "BIRA")
  new_model_file = os.getenv("NEW_MODEL_FILE", "True").lower() == "true"

  print(source_model_name, target_model_name, new_model_file)
  
  check_ollama_installed()
  
  # Check if target model is available
  if new_model_file or pull_source_model_if_needed(target_model_name, source_model_name):
    build_model_from_modelfile(target_model_name, source_model_name)

  if check_model_source(target_model_name, source_model_name):
        print("🦾 Model verification passed!")
  else:
      print("😭 Model verification failed!")
      exit(1)
    
  test_ollama_run(target_model_name)

main()