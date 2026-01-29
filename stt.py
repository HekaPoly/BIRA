import torch
import torchaudio

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimum.transformers import BetterTransformer

MODEL_ID = "distil-whisper/distil-large-v2"
FILE = "recording.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device: ", DEVICE)
waveform, sample_rate = torchaudio.load(FILE)
waveform = waveform.mean(dim=0)

processor = WhisperProcessor.from_pretained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to(DEVICE)

model = BetterTransform.transform(model, keep_original_model=False)
model.eval()

inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with torch.no_grad():
    generated_ids =  model.generate(**inputs, task="translate", language="fr")

text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(text)

