# coding=utf-8
# Expanded Custom UI for Qwen3-TTS
# Includes Voice Library, Clone & Generate, and Save/Load tabs without Japanese translations.

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

def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = x.astype(np.float32) / max(abs(info.min), info.max) if info.min < 0 else (x.astype(np.float32) - (info.max + 1) / 2.0) / ((info.max + 1) / 2.0)
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6: y /= (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip: y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1: y = np.mean(y, axis=-1).astype(np.float32)
    return y

def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None: return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        return _normalize_audio(audio["data"]), int(audio["sampling_rate"])
    return None

# --- Core UI Logic ---

def build_custom_ui(tts: Qwen3TTSModel):
    supported_langs_raw = []
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_langs_raw = tts.model.get_supported_languages()
    
    lang_choices_disp, lang_map = _build_choices_and_map([x for x in (supported_langs_raw or [])])

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
        primary_hue="blue",
        secondary_hue="slate",
    )

    css = ".gradio-container {max-width: 1100px !important; margin: auto;}"

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Custom Interface") as demo:
        gr.Markdown("# 🎙️ Qwen3-TTS Expanded Interface")

        with gr.Tabs():
            # --- TAB 1: Voice Library ---
            with gr.Tab("Voice Library"):
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
                        lib_text = gr.Textbox(label="Text", lines=4, placeholder="Enter text to synthesize...")
                        with gr.Row():
                            lib_lang = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Japanese")
                            lib_instruct = gr.Textbox(label="Instruction", placeholder="e.g. Speak faster")
                        lib_gen_btn = gr.Button("🚀 Generate Speech", variant="primary")

                with gr.Row():
                    lib_audio_out = gr.Audio(label="Result", type="numpy")
                    lib_status = gr.Textbox(label="Status", interactive=False)

            # --- TAB 2: Clone & Generate ---
            with gr.Tab("Clone & Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1. Reference Audio")
                        ref_audio = gr.Audio(label="Reference Audio")
                        ref_text = gr.Textbox(label="Reference Text", placeholder="Required for high quality cloning")
                        xvec_only = gr.Checkbox(label="Use x-vector only (Experimental)", value=False)

                    with gr.Column(scale=3):
                        gr.Markdown("### 2. Synthesis Settings")
                        clone_text_in = gr.Textbox(label="Target Text", lines=4, placeholder="Enter text to synthesize...")
                        with gr.Row():
                            clone_lang = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Japanese")
                            clone_instruct = gr.Textbox(label="Instruction", placeholder="e.g. Say it with emotion")
                        clone_btn = gr.Button("🧬 Clone & Generate", variant="primary")

                with gr.Row():
                    clone_audio_out = gr.Audio(label="Result", type="numpy")
                    clone_status = gr.Textbox(label="Status", interactive=False)

            # --- TAB 3: Save / Load ---
            with gr.Tab("Save & Load"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 💾 Save Voice Profile")
                        save_ref_audio = gr.Audio(label="Reference Audio", type="numpy")
                        save_ref_text = gr.Textbox(label="Reference Text")
                        save_xvec_only = gr.Checkbox(label="Use x-vector only", value=False)
                        voice_name_in = gr.Textbox(label="Save Name", placeholder="e.g. curious_voice")
                        save_btn = gr.Button("📥 Analyze & Save Voice", variant="primary")
                        save_file_out = gr.File(label="Generated Profile (.pt)")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📁 Load & Generate")
                        load_file_in = gr.File(label="Upload .pt file")
                        load_text = gr.Textbox(label="Target Text", lines=4)
                        with gr.Row():
                            load_lang = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Japanese")
                            load_instruct = gr.Textbox(label="Instruction")
                        load_btn = gr.Button("🚀 Generate from File", variant="primary")

                with gr.Row():
                    load_audio_out = gr.Audio(label="Result", type="numpy")
                    load_status = gr.Textbox(label="Status", interactive=False)

        # --- Event Handlers ---

        def run_lib_gen(local_voice, text, lang_disp, instruct):
            try:
                if not local_voice or not text.strip(): return None, "❌ Missing voice selection or text."
                path = os.path.join(VOICES_DIR, local_voice)
                payload = torch.load(path, map_location="cpu", weights_only=True)
                items: List[VoiceClonePromptItem] = []
                for d in payload["items"]:
                    rc = d.get("ref_code"); rs = d.get("ref_spk_embedding")
                    items.append(VoiceClonePromptItem(
                        ref_code=torch.tensor(rc) if rc is not None else None,
                        ref_spk_embedding=torch.tensor(rs) if rs is not None else None,
                        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                        icl_mode=bool(d.get("icl_mode", True)),
                        ref_text=d.get("ref_text"),
                    ))
                l = lang_map.get(lang_disp, "Auto")
                wavs, sr = tts.generate_voice_clone(text=text.strip(), language=l, voice_clone_prompt=items, instruct=instruct or None)
                return _wav_to_gradio_audio(wavs[0], sr), "✅ Success"
            except Exception as e: return None, f"⚠️ Error: {e}"

        def run_voice_clone(aud, txt, use_xvec, text, lang_disp, instruct):
            try:
                if not text.strip() or aud is None: return None, "❌ Missing audio or target text."
                at = _audio_to_tuple(aud)
                l = lang_map.get(lang_disp, "Auto")
                wavs, sr = tts.generate_voice_clone(text=text.strip(), language=l, ref_audio=at, ref_text=txt.strip() or None, x_vector_only_mode=use_xvec, instruct=instruct or None)
                return _wav_to_gradio_audio(wavs[0], sr), "✅ Success"
            except Exception as e: return None, f"⚠️ Error: {e}"

        def save_voice(aud, txt, use_xvec, name):
            try:
                at = _audio_to_tuple(aud)
                if at is None: return None, "❌ No audio data."
                items = tts.create_voice_clone_prompt(ref_audio=at, ref_text=txt.strip() or None, x_vector_only_mode=use_xvec)
                filename = (name.strip() or "unnamed_voice") + ".pt"
                out_path = os.path.join(VOICES_DIR, filename)
                torch.save({"items": [asdict(it) for it in items]}, out_path)
                return out_path, f"✅ Saved to {out_path}"
            except Exception as e: return None, f"⚠️ Error: {e}"

        def load_and_gen(file_obj, text, lang_disp, instruct):
            try:
                if not file_obj or not text.strip(): return None, "❌ Missing file or text."
                path = getattr(file_obj, "name", str(file_obj))
                payload = torch.load(path, map_location="cpu", weights_only=True)
                items = [VoiceClonePromptItem(
                    ref_code=torch.tensor(d["ref_code"]) if d.get("ref_code") is not None else None,
                    ref_spk_embedding=torch.tensor(d["ref_spk_embedding"]) if d.get("ref_spk_embedding") is not None else None,
                    x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                    icl_mode=bool(d.get("icl_mode", True)),
                    ref_text=d.get("ref_text")
                ) for d in payload["items"]]
                wavs, sr = tts.generate_voice_clone(text=text.strip(), language=lang_map.get(lang_disp, "Auto"), voice_clone_prompt=items, instruct=instruct or None)
                return _wav_to_gradio_audio(wavs[0], sr), "✅ Success"
            except Exception as e: return None, f"⚠️ Error: {e}"

        def refresh_lib():
            v = get_local_voices(); return gr.update(choices=v, value=v[0] if v else None)

        lib_gen_btn.click(run_lib_gen, [local_voice_dropdown, lib_text, lib_lang, lib_instruct], [lib_audio_out, lib_status])
        refresh_btn.click(refresh_lib, [], [local_voice_dropdown])
        clone_btn.click(run_voice_clone, [ref_audio, ref_text, xvec_only, clone_text_in, clone_lang, clone_instruct], [clone_audio_out, clone_status])
        save_btn.click(save_voice, [save_ref_audio, save_ref_text, save_xvec_only, voice_name_in], [save_file_out, load_status])
        load_btn.click(load_and_gen, [load_file_in, load_text, load_lang, load_instruct], [load_audio_out, load_status])

    return demo

def main():
    parser = argparse.ArgumentParser(description="Expanded Qwen3-TTS Interface")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ssl-certfile", default=None)
    parser.add_argument("--ssl-keyfile", default=None)
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"Loading {args.checkpoint} on {args.device} ({args.dtype})...")
    tts = Qwen3TTSModel.from_pretrained(args.checkpoint, device_map=args.device, dtype=dtype, attn_implementation="flash_attention_2" if args.flash_attn else None)
    
    demo = build_custom_ui(tts)
    demo.launch(server_name=args.ip, server_port=args.port, ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile, ssl_verify=not args.ssl_certfile)

if __name__ == "__main__":
    main()
