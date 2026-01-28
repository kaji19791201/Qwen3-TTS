# coding=utf-8
# Custom Local Voice Library UI for Qwen3-TTS
# This script is a standalone Gradio app designed to manage and use local voice profiles.

import os
import glob
import torch
import gradio as gr
import numpy as np
import argparse
from dataclasses import asdict
from typing import List, Optional, Tuple, Any, Dict
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# --- Constants & Helper Functions ---
VOICES_DIR = "./data/voices"
DEFAULT_MODEL = "./data/checkpoints/Qwen3-TTS-12Hz-1.7B-Base"

def get_local_voices(directory=VOICES_DIR):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return []
    files = glob.glob(os.path.join(directory, "*.pt"))
    return sorted([os.path.basename(f) for f in files])

def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])

def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping

def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav

# --- Core UI Logic ---

def build_voice_library_ui(tts: Qwen3TTSModel):
    # Get initial data from model
    supported_langs_raw = []
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_langs_raw = tts.model.get_supported_languages()
    
    lang_choices_disp, lang_map = _build_choices_and_map([x for x in (supported_langs_raw or [])])

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
        primary_hue="blue",
        secondary_hue="slate",
    )

    css = ".gradio-container {max-width: 1000px !important; margin: auto;}"

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Local Voice Library") as demo:
        gr.Markdown(
            f"""
# 🎙️ Qwen3-TTS Local Voice Library
Manage and use your saved voice profiles.
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 1. Select Voice Profile")
                local_voices_list = get_local_voices()
                local_voice_dropdown = gr.Dropdown(
                    label="Saved Voices (.pt)",
                    choices=local_voices_list,
                    value=local_voices_list[0] if local_voices_list else None,
                    interactive=True,
                )
                refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")

            with gr.Column(scale=3):
                gr.Markdown("### 2. Synthesis Settings")
                target_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=4,
                    placeholder="Enter the text you want the voice to say...",
                )
                with gr.Row():
                    lang_in = gr.Dropdown(
                        label="Language",
                        choices=lang_choices_disp,
                        value="Japanese" if "Japanese" in lang_choices_disp else None,
                        interactive=True,
                    )
                    instruct_in = gr.Textbox(
                        label="Instruction (Optional)",
                        placeholder="e.g. Speak excitedly, with a deep voice.",
                    )
                
                gen_btn = gr.Button("🚀 Generate Speech", variant="primary")

        with gr.Row():
            with gr.Column():
                audio_out = gr.Audio(label="Resulting Audio", type="numpy")
                status_out = gr.Textbox(label="Status", interactive=False)

        # --- Event Handlers ---

        def run_lib_gen(local_voice, text: str, lang_disp: str, instruct: str):
            try:
                if not local_voice:
                    return None, "❌ Please select a voice profile."
                if not text or not text.strip():
                    return None, "❌ Please enter some text."
                
                path = os.path.join(VOICES_DIR, local_voice)
                payload = torch.load(path, map_location="cpu", weights_only=True)
                items_raw = payload["items"]
                items: List[VoiceClonePromptItem] = []
                for d in items_raw:
                    ref_code = d.get("ref_code", None)
                    if ref_code is not None and not torch.is_tensor(ref_code):
                        ref_code = torch.tensor(ref_code)
                    ref_spk = d.get("ref_spk_embedding", None)
                    if not torch.is_tensor(ref_spk):
                        ref_spk = torch.tensor(ref_spk)
                    items.append(
                        VoiceClonePromptItem(
                            ref_code=ref_code,
                            ref_spk_embedding=ref_spk,
                            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                            icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                            ref_text=d.get("ref_text", None),
                        )
                    )
                
                language = lang_map.get(lang_disp, "Auto")
                print(f"Synthesizing with voice: {local_voice}, lang: {language}")
                
                wavs, sr = tts.generate_voice_clone(
                    text=text.strip(),
                    language=language,
                    voice_clone_prompt=items,
                    instruct=(instruct or "").strip() or None,
                )
                return _wav_to_gradio_audio(wavs[0], sr), "✅ Finished successfully."
            except Exception as e:
                print(f"Error: {e}")
                return None, f"⚠️ Error: {str(e)}"

        def refresh_lib():
            voices = get_local_voices()
            return gr.update(choices=voices, value=voices[0] if voices else None)

        gen_btn.click(
            run_lib_gen, 
            inputs=[local_voice_dropdown, target_text, lang_in, instruct_in], 
            outputs=[audio_out, status_out]
        )
        refresh_btn.click(refresh_lib, inputs=[], outputs=[local_voice_dropdown])

    return demo

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Standalone Local Voice Library UI")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL, help="Model checkpoint path")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--port", type=int, default=8001, help="Gradio port")
    parser.add_argument("--ip", default="0.0.0.0", help="Gradio IP")
    
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} on {args.device}...")
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=torch.float32 if args.device == "mps" else torch.bfloat16,
    )

    demo = build_voice_library_ui(tts)
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == "__main__":
    main()
