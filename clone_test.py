import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os

# Set device
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32 # Use float32 on MPS for better compatibility
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}, dtype: {dtype}")

# Load the 0.6B Base model
# We point to the local directory where we downloaded the model
model_path = "./Qwen3-TTS-12Hz-0.6B-Base"
if not os.path.exists(model_path):
    print(f"Model path {model_path} not found. Using Hugging Face ID.")
    model_path = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map=device,
    dtype=dtype,
)

# For testing, we need a reference audio. 
# Since we don't have one from the user yet, let's look for a sample in the repo if it was cloned.
# Or we can use a small dummy noise if no audio is provided, but that won't show the cloning quality.
# Instead, I'll provide a placeholder path and ask the user.

ref_audio = "sample_voice.wav" # User should replace this with their wav file
ref_text = "Hello, this is my voice." # Transcript of the reference audio
gen_text = "こんにちは、Qwen3-TTSです。これはボイスクローニングのテストです。"

if os.path.exists(ref_audio):
    print(f"Running voice cloning with {ref_audio}")
    try:
        wavs, sr = model.generate_voice_clone(
            text=gen_text,
            ref_audio=ref_audio,
            ref_text=ref_text
        )
        output_file = "output_cloned.wav"
        sf.write(output_file, wavs[0], sr)
        print(f"Voice cloning success! Saved to {output_file}")
    except Exception as e:
        print(f"Error during voice cloning: {e}")
else:
    print(f"Reference audio '{ref_audio}' not found. Please provide a WAV file to test voice cloning.")
    print("You can also try running the Web UI demo: qwen-tts-demo Qwen/Qwen3-TTS-12Hz-0.6B-Base")
