import os
import torch
import numpy as np
import re
import scipy.io.wavfile as wavfile
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

@dataclass
class NarrationChunk:
    text: str
    instruction: Optional[str] = None
    pause_duration: Optional[float] = None

class PodcastEngine:
    def __init__(self, tts_model: Qwen3TTSModel, voices_dir: str = "./data/voices"):
        self.tts = tts_model
        self.voices_dir = voices_dir
        self.voice_cache = {}

    def chunk_text(self, text: str, max_chars: int = 500) -> List[NarrationChunk]:
        """
        Splits long text into smaller chunks for processing.
        Skips metadata headers. Ideally keeps larger blocks for better TTS flow.
        """
        lines = [line.strip() for line in text.splitlines()]
        chunks = []
        
        # Metadata pattern
        meta_pattern = re.compile(r'^(\[File|Date:|Location:)', re.IGNORECASE)

        current_block = ""

        for line in lines:
            # Skip metadata lines
            if meta_pattern.search(line):
                continue
            
            # If empty line, it might be a paragraph break. 
            # If current block is long enough, push it.
            if not line:
                if current_block:
                    chunks.append(NarrationChunk(text=current_block.strip()))
                    current_block = ""
                continue

            # Add line to current block
            if current_block:
                current_block += "\n" + line
            else:
                current_block = line

            # If block gets too long, force split (roughly)
            # This is a fallback; ideally we split on newlines (paragraphs)
            if len(current_block) > max_chars:
                chunks.append(NarrationChunk(text=current_block.strip()))
                current_block = ""
        
        # Remaining text
        if current_block:
             chunks.append(NarrationChunk(text=current_block.strip()))
                
        return chunks

    def load_voice_profile(self, voice_name: str) -> List[VoiceClonePromptItem]:
        if voice_name in self.voice_cache:
            return self.voice_cache[voice_name]
        
        if not voice_name.endswith(".pt"):
            voice_path = os.path.join(self.voices_dir, voice_name + ".pt")
        else:
            voice_path = os.path.join(self.voices_dir, voice_name)

        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice profile not found: {voice_path}")

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
        self.voice_cache[voice_name] = items
        return items

    def normalize_audio(self, wav: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Simple peak normalization using numpy."""
        max_amplitude = np.max(np.abs(wav))
        if max_amplitude == 0:
            return wav
        
        # Convert dB to linear scale for amplitude
        target_amplitude = 10 ** (target_db / 20)
        return wav * (target_amplitude / max_amplitude)

    def generate_solo_podcast(
        self, 
        script_text: str, 
        voice_profile: str, 
        language: str = "Japanese",
        pause_duration: float = 0.6,
        cache_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generates a solo podcast from script text with optional caching.
        """
        chunks = self.chunk_text(script_text)
        voice_prompt = self.load_voice_profile(voice_profile)
        
        all_wavs = []
        sampling_rate = None

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        import hashlib

        for i, chunk in enumerate(chunks):
            if not chunk.text:
                continue
                
            # Create a hash for the chunk text to detect changes
            chunk_hash = hashlib.md5(f"{voice_profile}_{chunk.text}".encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"chunk_{i:03d}_{chunk_hash}.wav") if cache_dir else None
            
            wav_data = None
            sr = None

            if cache_file and os.path.exists(cache_file):
                print(f"Using cached chunk {i+1}/{len(chunks)}...")
                sr, wav_data = wavfile.read(cache_file)
                # Convert back to float32
                wav_data = wav_data.astype(np.float32) / 32767.0
            else:
                print(f"Generating chunk {i+1}/{len(chunks)}: {chunk.text[:30]}...")
                wavs, sr_gen = self.tts.generate_voice_clone(
                    text=chunk.text,
                    language=language,
                    voice_clone_prompt=voice_prompt
                )
                wav_data = wavs[0]
                sr = sr_gen
                
                if cache_file:
                    wavfile.write(cache_file, sr, (wav_data * 32767).astype(np.int16))
            
            if sampling_rate is None:
                sampling_rate = sr
            
            all_wavs.append(wav_data)
            
            # Add default pause between chunks
            if pause_duration > 0 and i < len(chunks) - 1:
                silence = np.zeros(int(sampling_rate * pause_duration), dtype=np.float32)
                all_wavs.append(silence)

        if not all_wavs:
            return np.array([], dtype=np.float32), 24000

        combined_wav = np.concatenate(all_wavs)
        normalized_wav = self.normalize_audio(combined_wav)
        
        return normalized_wav, sampling_rate

if __name__ == "__main__":
    pass
