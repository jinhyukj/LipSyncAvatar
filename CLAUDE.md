# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveAvatar is a real-time, streaming, infinite-length audio-driven avatar video generation framework. It uses a 14B-parameter diffusion model (WAN 2.2) with LoRA fine-tuning to generate lip-synced talking avatar videos from a reference image + audio input. Achieves 20 FPS on 5×H800 GPUs with 4-step sampling.

**Origin**: Alibaba Quark team, forked/adapted for LipSync work.

## Common Commands

### Environment Setup
```bash
conda create -n liveavatar python=3.10 -y && conda activate liveavatar
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
apt-get update && apt-get install -y ffmpeg
```

### Download Checkpoints
```bash
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./ckpt/Wan2.2-S2V-14B
huggingface-cli download Quark-Vision/Live-Avatar --local-dir ./ckpt/LiveAvatar
```

### Inference (CLI)
```bash
# Single GPU (80GB VRAM, offline generation, CPU offloading enabled)
bash infinite_inference_single_gpu.sh

# Multi-GPU (5× GPUs, real-time with TPP pipeline parallelism)
bash infinite_inference_multi_gpu.sh
```

### Gradio Web UI
```bash
bash gradio_single_gpu.sh    # Single GPU, port 7860
bash gradio_multi_gpu.sh     # 5 GPUs with TPP, port 7860
```

### Running Inference Directly
All inference uses `torchrun`. Single GPU uses `--nproc_per_node=1 --single_gpu`, multi-GPU uses `--nproc_per_node=5` with `--enable_vae_parallel` and `--num_gpus_dit 4`.

```bash
torchrun --nproc_per_node=1 --master_port=29102 minimal_inference/s2v_streaming_interact.py \
    --task s2v-14B --size "704*384" --sample_steps 4 --single_gpu \
    --training_config liveavatar/configs/s2v_causal_sft.yaml \
    --ckpt_dir ckpt/Wan2.2-S2V-14B/ --load_lora --lora_path_dmd "Quark-Vision/Live-Avatar" \
    --image examples/dwarven_blacksmith.jpg --audio examples/dwarven_blacksmith.wav \
    --prompt "description..." --offload_model True --convert_model_dtype --infer_frames 48
```

## Architecture

### Entry Points
- `minimal_inference/s2v_streaming_interact.py` — CLI inference (block-wise autoregressive video generation)
- `minimal_inference/gradio_app.py` — Gradio Web UI with multi-GPU worker loop pattern
- `minimal_inference/batch_eval.py` — Batch evaluation

### Pipeline Selection (Single vs Multi-GPU)
The entry points dynamically import different pipeline classes based on world_size:
- **Single GPU**: `liveavatar.models.wan.causal_s2v_pipeline.WanS2V` — sequential processing with CPU offloading
- **Multi-GPU (5)**: `liveavatar.models.wan.causal_s2v_pipeline_tpp.WanS2V` — Timestep-forcing Pipeline Parallelism (4 DiT GPUs + 1 VAE GPU)

Multi-GPU requires exactly 5 GPUs: 4 for DiT pipeline parallelism + 1 for VAE parallel decoding.

### Core Model Stack
```
Audio (wav) ──→ Audio Encoder (conformer) ──→ Audio embeddings ─┐
Text prompt ──→ T5 Text Encoder ──→ Text embeddings ────────────┤
Reference image ──→ VAE Encoder ──→ Appearance features ────────┤
                                                                 ▼
                                        CausalWanModel_S2V (14B DiT)
                                        (4-step flow matching diffusion)
                                                                 │
                                                                 ▼
                                            VAE Decoder ──→ RGB video frames
                                                                 │
                                                                 ▼
                                            merge_video_audio ──→ MP4 output
```

### Key Model Files
- `liveavatar/models/model_interface.py` — Abstract base classes (DiffusionModel, VAE, TextEncoder)
- `liveavatar/models/wan/causal_model_s2v.py` — The 14B DiT model with KV-cache for causal generation (~1574 lines)
- `liveavatar/models/wan/causal_s2v_pipeline.py` — Single-GPU S2V pipeline (~1223 lines)
- `liveavatar/models/wan/causal_s2v_pipeline_tpp.py` — Multi-GPU TPP pipeline (~1181 lines)
- `liveavatar/models/wan/causal_audio_encoder.py` — Audio encoding for lip-sync
- `liveavatar/models/wan/causal_motioner.py` — Motion generation module
- `liveavatar/models/wan/flow_match.py` — Flow matching diffusion logic

### WAN 2.2 Base Model (`liveavatar/models/wan/wan_2_2/`)
- `configs/` — Model configs for different task variants (T2V, I2V, S2V, TI2V)
- `modules/model.py` — Main DiT architecture
- `modules/attention.py` — Attention mechanisms
- `modules/s2v/` — Speech-to-Video specific modules (audio_encoder, motioner, model_s2v)
- `modules/vae2_2.py` — VAE decoder
- `modules/t5.py` — T5 text encoder
- `distributed/` — FSDP, sequence parallelism, Ulysses parallelism utilities
- `utils/fm_solvers.py` — Flow matching ODE solvers (Euler, DPM++, UniPC)

### Configuration
- `liveavatar/configs/s2v_causal_sft.yaml` — Training/LoRA config (LoRA rank=128, alpha=64, targets: q,k,v,o,ffn.0,ffn.2)
- `liveavatar/utils/args_config.py` — Argument parsing and config loading

### Utilities
- `liveavatar/utils/model_manager.py` — Model download and checkpoint management
- `liveavatar/utils/load_weight_utils.py` — Safetensors weight loading, LoRA injection
- `liveavatar/utils/io_utils.py` — Video/audio I/O with ffmpeg
- `liveavatar/utils/router/synthesize_audio.py` — Audio merging
- `liveavatar/utils/sync_net/` — Audio-visual synchronization scoring

## Key Parameters

| Parameter | Single GPU | Multi-GPU |
|-----------|-----------|-----------|
| `--size` | `704*384` | `720*400` |
| `--num_gpus_dit` | 1 | 4 |
| `--offload_model` | True | False |
| `--enable_vae_parallel` | (not used) | required |
| `--single_gpu` | required | (not used) |
| `--sample_steps` | 4 | 4 |
| `--infer_frames` | 48 | 48 |
| `--sample_solver` | euler | euler |
| `--sample_guide_scale` | 0 | 0 |

- `--num_clip` controls how many 48-frame clips to generate (up to 10000 for infinite video)
- `--size` represents area; aspect ratio follows the input image
- `--infer_frames` must be a multiple of 4
- `--enable_online_decode` is off by default for single GPU to avoid CPU offloading overhead; enable it for better quality on very long videos

## Notes
- Python 3.10, PyTorch 2.8.0, CUDA 12.8, flash-attn 2.8.3
- Minimum 80GB VRAM per GPU
- Output saved to `./output/` by default
- The training config YAML contains Chinese comments (this is expected)
- LoRA weights can be loaded from HuggingFace hub paths (e.g., `Quark-Vision/Live-Avatar`) or local `.pt` files
- No test suite exists; validation is done via `batch_eval.py` and sync_net metrics
