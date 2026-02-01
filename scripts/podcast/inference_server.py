import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import base64
import io
import scipy.io.wavfile as wavfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

app = FastAPI(title="Qwen3-TTS Inference API")

# Global model instance
model = None
voice_prompts = {}

class GenerateRequest(BaseModel):
    text: str
    voice_profile_path: str
    language: str = "Japanese"
    temperature: float = 0.9

class GenerateResponse(BaseModel):
    audio_base64: str
    sample_rate: int

def load_voice_profile(voice_path: str):
    if voice_path in voice_prompts:
        return voice_prompts[voice_path]
    
    print(f"Loading voice profile: {voice_path}")
    payload = torch.load(voice_path, map_location="cpu", weights_only=True)
    items = []
    for d in payload["items"]:
        items.append(VoiceClonePromptItem(
            ref_code=torch.tensor(d["ref_code"]) if d.get("ref_code") is not None else None,
            ref_spk_embedding=torch.tensor(d["ref_spk_embedding"]) if d.get("ref_spk_embedding") is not None else None,
            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
            icl_mode=bool(d.get("icl_mode", True)),
            ref_text=d.get("ref_text"),
        ))
    voice_prompts[voice_path] = items
    return items

@app.on_event("startup")
async def startup_event():
    global model
    model_path = "./data/checkpoints/Qwen3-TTS-12Hz-1.7B-Base"
    print(f"Loading model from {model_path}...")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="mps",
        dtype=torch.float16
    )
    print("Model loaded successfully.")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        voice_prompt = load_voice_profile(request.voice_profile_path)
        
        wavs, sr = model.generate_voice_clone(
            text=request.text,
            language=request.language,
            voice_clone_prompt=voice_prompt,
            temperature=request.temperature
        )
        
        # Convert to base64
        audio_data = wavs[0]
        byte_io = io.BytesIO()
        wavfile.write(byte_io, sr, (audio_data * 32767).astype(np.int16))
        audio_base64 = base64.b64encode(byte_io.getvalue()).decode("utf-8")
        
        return GenerateResponse(audio_base64=audio_base64, sample_rate=sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
