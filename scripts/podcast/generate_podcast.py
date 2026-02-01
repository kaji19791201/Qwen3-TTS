import os
import sys
import argparse
import torch
import scipy.io.wavfile as wavfile

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qwen_tts import Qwen3TTSModel
from scripts.podcast.podcast_engine import PodcastEngine

# Default paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "data/checkpoints/Qwen3-TTS-12Hz-1.7B-Base")
VOICES_DIR = os.path.join(PROJECT_ROOT, "data/voices")

def main():
    parser = argparse.ArgumentParser(description="Generate a solo podcast from a script.")
    parser.add_argument("script_path", help="Path to the script file (.md or .txt)")
    parser.add_argument("--voice", required=True, help="Name of the voice profile (e.g. 'host1')")
    parser.add_argument("--output", default="output_podcast.wav", help="Output WAV filename")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL, help="Model checkpoint path")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--language", default="Japanese", choices=["Japanese", "English", "Chinese", "Korean"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()

    if not os.path.exists(args.script_path):
        print(f"Error: Script file not found: {args.script_path}")
        return

    # Load Script
    with open(args.script_path, "r", encoding="utf-8") as f:
        script_text = f.read()

    # Load Model
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"Loading model from {args.checkpoint} on {args.device}...")
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint, 
        device_map=args.device, 
        dtype=dtype, 
        attn_implementation="flash_attention_2" if args.flash_attn else None
    )

    # Initialize Engine
    engine = PodcastEngine(tts, voices_dir=VOICES_DIR)

    # Cache directory based on script name
    script_base = os.path.splitext(os.path.basename(args.script_path))[0]
    cache_dir = os.path.join(PROJECT_ROOT, "data/cache", script_base)

    # Generate Podcast
    print(f"Generating podcast for voice: {args.voice} (Incremental: ON)...")
    wav, sr = engine.generate_solo_podcast(
        script_text=script_text,
        voice_profile=args.voice,
        language=args.language,
        cache_dir=cache_dir
    )

    # Save Output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    wavfile.write(args.output, sr, wav)
    print(f"✅ Podcast saved to: {args.output}")

if __name__ == "__main__":
    main()
