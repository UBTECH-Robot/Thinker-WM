#!/usr/bin/env bash
# ThinkerWM one-click example: runs wm (with depth output) / wam / vla on built-in samples,
# outputs to outputs/example/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SAMPLE_DIR="examples/ewmbench_sample"
OUT_DIR="outputs/example"

# ---- Weight readiness check: prints download instructions and exits if any required file is missing ----
REQUIRED_FILES="
model_weights/checkpoint/pytorch_model_0.bin
model_weights/wan/config.json
model_weights/wan/Wan2.2_VAE.pth
model_weights/qwen3_vl/config.json
model_weights/action_stats.json
"
MISSING=""
for f in $REQUIRED_FILES; do
    if [ ! -f "$f" ]; then
        MISSING="$MISSING $f"
    fi
done
if [ -n "$MISSING" ]; then
    echo "ERROR: model_weights/ not ready, missing:$MISSING"
    echo ""
    echo "Please download according to the README 'Weights' section, place in this layout:"
    cat <<'LAYOUT'
model_weights/
├── checkpoint/pytorch_model_0.bin   # Main weights (~24 GB)
├── wan/config.json                  # Wan architecture config
├── wan/Wan2.2_VAE.pth               # Video VAE (not in checkpoint, download separately)
├── qwen3_vl/                        # Qwen3-VL config/tokenizer (VLM weights already in ckpt)
│   ├── config.json
│   ├── tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
│   ├── preprocessor_config.json / video_preprocessor_config.json
│   └── chat_template.json
└── action_stats.json                # Per-robot action normalization statistics
LAYOUT
    echo ""
    echo "Download link (replace with actual URL on release): https://huggingface.co/<your-org>/thinkerwm"
    exit 1
fi

# ---- Placeholder depth input for wm mode: codec domain [0,1] grayscale .npy (replace with sensor depth in production) ----
mkdir -p "$OUT_DIR"
python - <<'PY'
import os
import numpy as np
h, w = 480, 640
yy = np.linspace(0.0, 1.0, h)[:, None]
depth = (0.2 + 0.6 * np.broadcast_to(yy, (h, w))).astype(np.float32)
np.save(os.path.join("outputs", "example", "depth_first.npy"), depth)
PY

PROMPT="$(cat "$SAMPLE_DIR/prompt.txt")"

# 1) wm: first frame + prompt -> video (+ depth)
python main.py infer \
    --mode wm \
    --prompt "$PROMPT" \
    --first-frame "$SAMPLE_DIR/first_frame.png" \
    --depth-first "$OUT_DIR/depth_first.npy" \
    --output "$OUT_DIR/wm"

# 2) wam: prompt + first frame + state + robot -> video + actions
python main.py infer \
    --mode wam \
    --prompt "$PROMPT" \
    --first-frame "$SAMPLE_DIR/first_frame.png" \
    --state "$SAMPLE_DIR/state.npy" \
    --robot-name agibot-a2-d \
    --output "$OUT_DIR/wam"

# 3) vla: prompt + first frame + state + robot -> action sequence
python main.py infer \
    --mode vla \
    --prompt "$PROMPT" \
    --first-frame "$SAMPLE_DIR/first_frame.png" \
    --state "$SAMPLE_DIR/state.npy" \
    --robot-name agibot-a2-d \
    --output "$OUT_DIR/vla"

echo "done: outputs/example/{wm,wam,vla}"
