# EgoLAM
EgoLAM extracts a continuous 128-dimensional latent action from two RGB frames. A frozen DINOv2 ViT-B/14 feeds an eight-slot object-centric encoder (slot width 256). The inverse dynamics model pools the slots around each hand. Hand and camera annotations supervise training; extraction takes only the images.

[Model weights on Hugging Face](https://huggingface.co/dzyy123/EgoLAM) · [EgoVerse dataset](https://egoverse.ai/) · [LARYBench](https://github.com/meituan-longcat/LARYBench)

## Load the released model

From this repository, use Python 3.11 and a matched PyTorch/torchvision installation. Install the other dependencies with `pip install -r requirements.txt`. Authenticate with the Hub using an account that has access to the model. Download the two learned modules and the exact visual backbone:

```bash
hf download dzyy123/EgoLAM --local-dir weights/EgoLAM
hf download timm/vit_base_patch14_dinov2.lvd142m model.safetensors \
  --revision 4685c99dabffe5affac90bd99dbffd25801ae58d --local-dir weights/dino
export MODEL_DIR="$PWD/weights/EgoLAM"
export EGOLAM_BACKBONE="$PWD/weights/dino/model.safetensors"
```

```python
import os
import torch
from egolam.inference import load_models, encode
from evaluation.pairs import image

oc, lam = load_models(os.environ["MODEL_DIR"], device="cuda")
frames = torch.stack([image("earlier.jpg"), image("later.jpg")])[None]
z = encode(oc, lam, frames)  # [batch, 1, 128]
```

The native preprocessing resizes RGB frames to 518 × 518 with cubic interpolation. Slot initialization consumes RNG; fix the seed and pair batching for comparisons. The release includes trained slot and LAM tensors, the architecture config, and action/camera normalization statistics. The upstream DINOv2 backbone SHA-256 is `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b`.

## Train on EgoVerse

The native frame grid uses one frame every 12 raw steps (0.4 s). The LAM uses pairs three stored frames apart (1.2 s). Set `RLDS`, `FRAMES`, `LABELS`, and `CACHE` to your paths. The two preparation scripts use TensorFlow's TFRecord reader; install TensorFlow in a separate preprocessing environment if needed. Existing matching frame exports and consolidated labels can be used directly.

```bash
python tools/export_training_frames.py --rlds-root "$RLDS" --output "$FRAMES"
python tools/build_frame_labels.py --rlds-root "$RLDS" --frames-root "$FRAMES" --out-root "$LABELS"
torchrun --standalone --nproc_per_node=4 -m slot.train_oc_egoverse \
  --frames-root "$FRAMES" --labels-root "$LABELS" --out runs/oc \
  --n-slots 8 --slot-dim 256 --img-size 518 --steps 60000 --sim-temp .075 --sim-weight .5
OC=runs/oc/oc_slots_k8_060000.pt
torchrun --standalone --nproc_per_node=4 -m slot.build_slot_cache \
  --ckpt "$OC" --frames-root "$FRAMES" --labels-root "$LABELS" --out "$CACHE"
python -m slot.merge_slot_cache "$CACHE" "$LABELS"
torchrun --standalone --nproc_per_node=4 -m egolam.train_frozen \
  --oc-ckpt "$OC" --cache "$CACHE" --frames-root "$FRAMES" --labels-root "$LABELS" \
  --out runs/lam --stats-from configs/train.json --n-slots 8 --batch-size 192 \
  --gap 3 --lr .0001 --wd .01 --num-train-steps 100005 --stop-at 56000
```

Check `missing: 0` in the cache metadata before LAM training. The second stage starts a new LAM, freezes the object-centric encoder, and uses global batch 768, 5,000 warmup steps, and a cosine schedule with a 0.02 floor. The 56k checkpoint precedes the end of that schedule. The example uses four local workers. Data release, split, worker order, and software versions affect reproduction.

## Evaluate the frozen encoder

Run from the repository root. Make the LARYBench checkout importable and install its benchmark dependencies. Prepare the official datasets and split CSVs according to LARYBench. Benchmark footage, labels, and prediction caches are external inputs.

```bash
export LARY_ROOT=/path/to/LARYBench
export PYTHONPATH="$PWD:$LARY_ROOT:$PYTHONPATH"
```

**Original LARYBench.** Extract train and validation splits with `python -m evaluation.extract_lary --csv INPUT.csv --dataset TASK --mode MODE --stride STRIDE --split train --output FEATURES`; repeat with `--split val`. Use the native LARYBench data paths for the split. Run `python -m evaluation.original.readout TASK --train TRAIN_FEATURES.csv --val VAL_FEATURES.csv --lary-root "$LARY_ROOT" --output NEW_RUN`. Classification uses video features for human_1st/robot_1st, 20 epochs and global batch 256. Regression uses image pairs, the absolute command target, 50 epochs and batch 256; strides for calvin/vlabench/robocoin/agibotbeta are 5/5/10/45. Provide the benchmark action statistics with `--stats-json` when using a dataset that supplies them.

**Wrist-action decoding and intervention consistency.** Use the original EgoVerse causal-pair and intervention CSVs and their images. Preparation verifies published-run cohort hashes and split membership. The `tools/build_egoverse_pairs.py` and `tools/build_intervention_set.py` utilities expose the underlying data-generation steps. For the recorded evaluation specify the original cohort CSVs.

```bash
python -m evaluation.wrist.causal_data prepare --lary-root evaluation/reference \
  --data /path/to/regression/egoverse --output cohort
python -m evaluation.wrist.causal_data extract --model EgoLAM --cohort cohort \
  --weights configs/released_weights.json --output features
python -m evaluation.wrist.causal --model EgoLAM --cohort cohort --features features \
  --weights configs/released_weights.json --lary-root evaluation/reference \
  --source evaluation/reference/regression_main.py --readout official-mlp --seed 42 --ranks 4 --output runs/ad42
```

Repeat MLP fits with seeds 43 and 44. Choose `--readout ridge` or `--readout ic` and a fresh output directory for the ridge and intervention measurements. The IC analysis uses 605 matched source pairs. The MLP protocol uses 20 epochs and global batch 2048.

The matched real-head-turn analysis consumes the seed-42 AD predictions. Run `python -m evaluation.head_turn --cohort cohort --pairs-root /path/to/regression/egoverse --readout runs/ad42 --split unseen --output runs/turns`. Repeat with `--split val` for the independent validation split.

**SSv2 composition and action effects.** The dataset builders and split validation are in `extension_v1`. Start from the versioned task manifest for the chosen track and locally accessible videos. Each clip yields eight codes from nine selected frames. Cache extraction follows the source ordering and seeds each clip by its manifest row.

```bash
python -m evaluation.extract_ssv2 --manifest TASK_MANIFEST --video-root VIDEO_ROOT --output CACHE_ROOT
torchrun --standalone --nproc_per_node=4 -m evaluation.wrist.classify train \
  --task composition --manifest TASK_MANIFEST --cache CACHE_ROOT/packed \
  --lary-root "$LARY_ROOT" --native-eval evaluation/reference/ssv2_native_eval.py \
  --validation DATASET_VALIDATION.json --seed 0 --output NEW_RUN
python -m evaluation.wrist.classify score --task composition --manifest TASK_MANIFEST \
  --cache CACHE_ROOT/packed --lary-root "$LARY_ROOT" --native-eval evaluation/reference/ssv2_native_eval.py \
  --validation DATASET_VALIDATION.json --seed 0 --output NEW_RUN
```

Use the effect manifest with `--task effect`. Train each track with seeds 0, 1 and 2 at global batch 2048. This code retains the LARY attentive probe, scheduler and dev-set selection. The pinned native probe source is from LARYBench commit `e7feaf1b72921ee2c34e489adb0f45faf356ecee`.

## Source acknowledgments

The slot modules adapt VideoSAUR and use DINOv2 features via timm. The frozen-probe evaluation uses LARYBench. Third-party license notices are in `notices/`. Model weights originate from the 56k LAM and 60k slot checkpoint; safetensors export was checked tensor-by-tensor against both originals.
