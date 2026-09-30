<div align="center">
     
<img src="assets/logo1.png" width="200" alt="Thinker-WM Logo">

# Thinker-WM

**Thinker Team, Ubtech**

<!-- Replace the <your-org> repo/project links and the arXiv id below with the
     real addresses when publishing. -->

<a href="https://github.com/<your-org>/thinkerwm"><img src="https://img.shields.io/badge/GitHub-Code-blue?logo=github" alt="GitHub"></a>
<a href="https://huggingface.co/UBTECH-Robotics/Thinker-WM/tree/main"><img src="https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface" alt="HuggingFace"></a>
<a href="https://thinkercosmos.ubtrobot.com/thinker-wm/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2609.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"></a>

</div>

Thinker-WM is a unified multimodal world model that jointly models and generates
RGB, depth, and action through a shared intermediate Transformer backbone with
modality-specific input and output experts. This repository is the official
inference release: load a checkpoint and run six inference modes on a single GPU.

<p align="center">
  <img src="assets/teaser.jpg" width="90%">
</p>

## Introduction

World models have emerged as a promising paradigm for embodied AI, yet existing
systems remain fragmented: most focus on RGB and action, while simulation and
robot control are typically addressed by separate models or frameworks.

Thinker-WM unifies multimodal representation learning while supporting both
world/data generation and direct action modeling for robot control within a
single framework. A shared backbone processes interleaved RGB, depth, action,
and language tokens with directed cross-modal attention; modality-specific
experts handle inputs and outputs. A probing framework over the shared
representations reveals rich, fine-grained action information, providing a
strong foundation for action modeling and generation.

Thinker-WM is pretrained on 20K hours of egocentric data and 10K hours of robot
data (≈4K hours with aligned depth supervision) and supports multiple generation
modes — image-to-video-action, text-to-video-action, and action-to-video —
covering applications from simulation and planning to real-robot control. It
ranks **1st** on three public benchmarks: **EWMBench**, **World Olympiad**, and
**World Arena**.

## Installation

- Python 3.10+
- NVIDIA CUDA GPU; `flash-attn` is a hard dependency (see `requirements.txt`;
  pre-built wheels available if you cannot compile)
- `pip install -r requirements.txt`

## Model Weights

Weights live under `model_weights/` in the repo root (gitignored):

| File | Description | Size |
|---|---|---|
| `checkpoint/pytorch_model_0.bin` | ThinkerWM main weights (Wan backbone / VLM / action branch included) | ~24 GB |
| `wan/config.json` | Wan architecture config (structure only) | small |
| `wan/Wan2.2_VAE.pth` | Wan2.2 video VAE (separate file — it is not contained in the main checkpoint, but ships with `model_weights/`) | GB-scale |
| `qwen3_vl/` | Qwen3-VL directory, **config/tokenizer/processor files only** (VLM weights are inside the checkpoint; no safetensors needed) | ~12 MB |
| `action_stats.json` | Per-robot action normalization statistics | ~10 KB |

### HuggingFace Download

> Replace `<your-org>/thinkerwm` with the real repo id on release:
> https://huggingface.co/<your-org>/thinkerwm

```bash
# placeholder — replace with the actual repo id on release
# huggingface-cli download <your-org>/thinkerwm --local-dir model_weights
```

## Quickstart

One-command example (runs `wm` with depth / `wam` / `vla` on the bundled sample,
into `outputs/example/`; prints a download guide if weights are missing):

```bash
bash examples/run_example.sh
```

CLI (`--config` / `--checkpoint` default to the layout above and can be omitted).
The primary example is **WAM mode** — joint video + action generation from a
prompt, a first frame, and a robot state:

```bash
python main.py infer --mode wam \
    --prompt "$(cat examples/ewmbench_sample/prompt.txt)" \
    --first-frame examples/ewmbench_sample/first_frame.png \
    --state examples/ewmbench_sample/state.npy \
    --robot-name agibot-a2-d \
    --output outputs/wam_demo
# -> outputs/wam_demo/video.mp4 + actions.npy

# add depth output (requires enable_depth: true):
python main.py infer --mode wam ... --depth-first first_depth.npy --output outputs/wam_depth
# -> video.mp4 + actions.npy + depth.mp4 + depth_m.npy (metric depth)

# wm: video only;  vla: actions only
python main.py infer --mode wm  --prompt "..." --first-frame first_frame.png --output outputs/wm_demo
python main.py infer --mode vla --prompt "..." --first-frame first_frame.png --robot-name agibot-a2-d --output outputs/vla_demo
```

Python API:

```python
from main import ThinkerWMPipeline

pipe = ThinkerWMPipeline("configs/inference_example.yaml",
                         "model_weights/checkpoint")
out = pipe.infer(
    mode="wam",
    prompt="Pick up the red block.",
    first_frame="frame0.png",          # path / PIL.Image / [3,H,W] tensor
    state="0.01,0.02,...",             # optional; see Six Modes
    robot_name="agibot_beta",          # aliases from the config are supported
    depth_first="depth0.npy",          # optional; provided -> also produces depth
)
out["video"]    # [3, T, H, W] float [0,1]
out["actions"]  # [num_video_frames*ratio, action_dim] float
out["depth"]    # [3, T, H, W] float [0,1] codec domain (optional)
pipe.save_outputs(out, "outputs/demo")
```

## Six Modes

| mode | inputs | outputs |
|---|---|---|
| `wam` | prompt + first_frame + robot (state optional) | video + actions (+depth) |
| `wm` | prompt + first_frame (robot/depth_first optional) | video (+depth) |
| `vla` | prompt + first_frame + robot (state optional) | actions |
| `a2v` | first_frame + actions + robot (state optional) | video |
| `v2a` | video (exactly 1+num_video_frames frames) + robot (state optional) | actions |
| `uncond` | first_frame + robot (state optional) | video + actions |

**state is optional in every mode.** The model predicts relative end-effector
actions: without `state`, `actions` contains relative quantities (position
delta + relative 6D rotation); when `state` is given, actions are restored to
absolute quantities using it as the anchor. `robot_name` is required for all
action-input/output modes (normalization and dim-map lookup).

`a2v` / `v2a` / `uncond` take no prompt (an empty-text constant condition is
used internally); `wam` / `wm` / `vla` use the prompt directly as the VLM text
condition.

## Inference Hyperparameters

Overridable via CLI flags (or `pipe.infer(...)` kwargs):

| Flag | Default | Description |
|---|---|---|
| `--num-inference-steps` | 50 | Video denoising steps |
| `--guidance-scale` | 5.0 | Video CFG scale (forced to 1.0 in `uncond` mode) |
| `--action-guidance-scale` | 1.0 | Action CFG scale (off by default) |
| `--action-num-inference-steps` | 10 | Action denoising steps |
| `--action-smooth-alpha` | 0.0 | Action temporal EMA smoothing |
| `--depth-smooth-alpha` | 0.0 | Depth temporal EMA smoothing |

## Input / Output

- **first_frame**: RGB image, resized to `video_height x video_width`.
- **state / actions**: raw physical quantities in the robot reader convention;
  normalization and unified-space scatter happen at the model boundary.
- **depth_first**: codec-domain [0,1] grayscale; `.npy` float (already encoded),
  `.npy` uint16 (millimetres, encoded with `depth_z_near/z_far/mode`), or a
  grayscale image.
- **video (`v2a`)**: video file or `[T,3,H,W]` tensor with exactly
  `1+num_video_frames` frames (first frame = conditioning frame).
- Outputs: `video.mp4` (5 fps), `actions.npy` (`[action_chunk_size, action_dim]`),
  `depth.mp4` / `depth_m.npy`.

## Citation

If you find Thinker-WM useful, please cite:

```bibtex
@article{thinkerwm2026,
  title={Thinker-WM: A Unified World Model with Shared Multimodal Representations for Embodied Intelligence},
  author={Thinker Team, Ubtech},
  year={2026}
}
```

## Contributors

**Core Authors**: Chen Liao, Jiyuan Wang, Xiongwei Zhao

**Authors**: Peng Gao, Yiming Ma, Daqin Luo, Xiangming Xu

**Project Leader**: Baiyu Pan

**Advisors**: Jichao Jiao, Hailin Shi
