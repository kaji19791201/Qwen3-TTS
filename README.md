# Qwen3-TTS (kaji19791201 Fork)

> [!NOTE]
> This is a personal fork for local environment optimization on **Apple Silicon Mac**.
> For the latest official documentation, models, and usage, please visit the original repository.

## 🔗 Reference
👉 **[Official QwenLM/Qwen3-TTS Repository](https://github.com/QwenLM/Qwen3-TTS)**

## 🚀 Quick Start (Apple Silicon Mac)

### Launch Custom GUI
Apple Silicon Mac (MPS) 用の最適化設定でカスタムGUIを起動するコマンド：

```fish
source .venv/bin/activate.fish
python custom_gui.py --dtype float16 --no-flash-attn --ssl-certfile cert.pem --ssl-keyfile key.pem --port 8000
```

## 🛠️ Fork Specific Changes
- **Isolated Git Identity**: `kaji19791201` (configured via local/home gitconfig).
- **Environment**: Isolated `.venv` (Python 3.12).
- **Custom UI**: `custom_gui.py` supporting Library, Clone, and Save/Load.
- **Data Path**: Models and voice profiles are stored in `./data/` (ignored by Git).