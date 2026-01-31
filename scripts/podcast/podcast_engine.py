import os
import torch
import numpy as np
import re
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

    def chunk_text(self, text: str, max_chars: int = 150) -> List[NarrationChunk]:
        """
        Splits long text into smaller chunks.
        Extracts inline instructions like "(Instruction: whisper)" or "[Instruction: happy]".
        Skips metadata headers and handles standalone instructions.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks = []
        
        # Regex to find instructions
        instr_pattern = re.compile(r'[\(\[]Instruction:\s*([^\)\]]+)[\)\]]', re.IGNORECASE)
        # Metadata pattern
        meta_pattern = re.compile(r'^(\[File|Date:|Location:)', re.IGNORECASE)

        pending_instruction = None

        for line in lines:
            # Skip metadata lines
            if meta_pattern.search(line):
                continue

            # Extract instruction if present
            current_instruction = None
            match = instr_pattern.search(line)
            if match:
                current_instruction = match.group(1).strip()
                # Remove the instruction tag from the text
                line = instr_pattern.sub('', line).strip()

            # If it's a standalone instruction (no text)
            if not line:
                if current_instruction:
                    # Check for explicit pause instruction (e.g., "pause 2s" or "2秒休止")
                    pause_match = re.search(r'pause\s*([\d\.]+)\s*s', current_instruction, re.I)
                    if pause_match:
                        chunks.append(NarrationChunk(text="", pause_duration=float(pause_match.group(1))))
                    else:
                        # For other standalone instructions (like "takes a deep breath")
                        # We might want a longer pause even if no text is generated
                        chunks.append(NarrationChunk(text="", instruction=current_instruction, pause_duration=1.2))
                        pending_instruction = current_instruction
                continue

            # Combine with pending instruction if any
            final_instruction = current_instruction or pending_instruction
            pending_instruction = None # Reset after applying

            # If line is short enough, keep it
            if len(line) <= max_chars:
                chunks.append(NarrationChunk(text=line, instruction=final_instruction))
                continue
            
            # Sub-split by sentence ending punctuation
            sub_lines = re.split(r'(?<=[。！？.!?.])', line)
            current_text = ""
            
            for sub in sub_lines:
                sub = sub.strip()
                if not sub: continue
                
                if len(current_text) + len(sub) <= max_chars:
                    current_text += sub
                else:
                    if current_text:
                        chunks.append(NarrationChunk(text=current_text, instruction=final_instruction))
                    current_text = sub
            
            if current_text:
                chunks.append(NarrationChunk(text=current_text, instruction=final_instruction))
                
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
        pause_duration: float = 0.6
    ) -> Tuple[np.ndarray, int]:
        """
        Generates a solo podcast from script text.
        """
        chunks = self.chunk_text(script_text)
        voice_prompt = self.load_voice_profile(voice_profile)
        
        all_wavs = []
        sampling_rate = None

        for i, chunk in enumerate(chunks):
            if chunk.text:
                print(f"Generating chunk {i+1}/{len(chunks)}: {chunk.text[:30]}...")
                wavs, sr = self.tts.generate_voice_clone(
                    text=chunk.text,
                    language=language,
                    instruct=chunk.instruction,
                    voice_clone_prompt=voice_prompt
                )
                
                if sampling_rate is None:
                    sampling_rate = sr
                
                all_wavs.append(wavs[0])
            
            # Add pause after this chunk (or as its only content)
            actual_pause = chunk.pause_duration if chunk.pause_duration is not None else pause_duration
            if actual_pause > 0:
                if sampling_rate is None: sampling_rate = 24000 # Fallback
                silence = np.zeros(int(sampling_rate * actual_pause), dtype=np.float32)
                all_wavs.append(silence)

        if not all_wavs:
            return np.array([], dtype=np.float32), 24000

        combined_wav = np.concatenate(all_wavs)
        normalized_wav = self.normalize_audio(combined_wav)
        
        return normalized_wav, sampling_rate

if __name__ == "__main__":
    pass
