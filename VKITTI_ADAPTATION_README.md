# VKITTI Car24 Adaptation Notes

Reference upstream repo:
https://github.com/JagritiSahi/Multi-Camera-feature-for-Robust-SF.git

This repo was adapted from the original CMU Panoptic / Shelf / Campus human
multi-view setup to VKITTI stereo car keypoints. The model is still RAFT3D +
MVGFormer-style sparse keypoint scene-flow training, but the dataset source is
now VKITTI car annotations instead of human joints.

## Current Dataset Counts

The active training manifest is:

```text
/home/users/multicog/saksham_1/new_proj/data/Vkitti/vkitti_car24_world_all_scenes.jsonl
```

Current split:

| split | scenes | frame-level stereo samples |
|---|---|---:|
| train | Scene01, Scene02, Scene06, Scene20 | 8421 |
| validation/test | Scene18 | 1492 |

Train scene counts:

| scene | samples |
|---|---:|
| Scene01 | 2104 |
| Scene02 | 1085 |
| Scene06 | 1052 |
| Scene20 | 4180 |

## Expected VKITTI Data Layout

Use one root, preferably via `DATASET.ROOT` in config or `VKITTI_ROOT`:

```text
$VKITTI_ROOT/
  vkitti_car24_world_all_scenes.jsonl
  scene01_car24_world_annotations.jsonl
  scene01_car24_world_summary.csv
  ...
  vkitti_2.0.3_rgb/
  vkitti_2.0.3_depth/
  vkitti_2.0.3_forwardFlow/
  vkitti_2.0.3_forwardSceneFlow/
  vkitti_2.0.3_textgt/
```

The textgt folders provide camera and object metadata:

```text
vkitti_2.0.3_textgt/SceneXX/<variation>/
  intrinsic.txt
  extrinsic.txt
  pose.txt
  bbox.txt
```

## Annotation Format

The VKITTI annotation is frame-level and car-Panoptic-like:

```json
{
  "scene": "Scene20",
  "variation": "clone",
  "frame": 30,
  "camera": "Camera_0",
  "cars": [
    {
      "id": 2,
      "joints24": [[1.2, 0.3, 12.4, 0.91]],
      "sources": ["direct"]
    }
  ]
}
```

Rules:

- One JSONL row is one `scene + variation + frame + camera`.
- `cars` contains all accepted cars in that frame/camera.
- Each car has 24 keypoints.
- `joints24` is `(24, 4)`: `[world_x, world_y, world_z, confidence]`.
- Missing keypoint is `[0, 0, 0, 0]`.
- `sources` is length 24, e.g. `direct`, `pose_fill_cross_variation`, `missing`.
- Coordinates are world coordinates in meters, not pixels.

## Core Files Changed

### `MVG_SF_V1_Shelf_push/lib/dataset/vkitti_keypoints_raft.py`

This is the VKITTI dataset adapter. It replaces the original human-dataset
loader for this experiment.

Main responsibilities:

- Reads `vkitti_car24_world_all_scenes.jsonl`.
- Uses scene split:
  - train: `Scene01`, `Scene02`, `Scene06`, `Scene20`
  - validation/test: `Scene18`
- Requires stereo samples with both `Camera_0` and `Camera_1`.
- Rejects samples missing required temporal files:
  - RGB at `t`
  - RGB at `t+1`
  - depth at `t`
  - depth at `t+1`
  - forward optical flow
  - forward scene flow
- Projects world keypoints into image pixels using VKITTI intrinsics/extrinsics.
- Loads sparse supervision only at valid keypoint pixels.
- Builds Panoptic-like fields expected downstream:
  - `joints`
  - `joints_3d`
  - `joints_vis`
  - `keypoint_confidence`
  - `track_ids`
  - `num_person`
  - `camera`
  - `camera_Intri`
  - `camera_Extri`

### `MVG_SF_V1_Shelf_push/configs/vkitti/vkitti_keypoints_raft.yaml`

This config selects the VKITTI adapter.

Important fields:

```yaml
DATASET:
  TRAIN_DATASET: vkitti_keypoints_raft
  TEST_DATASET: vkitti_keypoints_raft
  ROOT: /home/users/multicog/saksham_1/new_proj/data/Vkitti
  CAMERA_NUM: 2
  TRAIN_SUBSET: train
  TEST_SUBSET: validation

NETWORK:
  IMAGE_SIZE: [1242, 375]
  NUM_JOINTS: 24

MULTI_PERSON:
  MAX_PEOPLE_NUM: 64

DECODER:
  t_pose_dir: ./data/vkitti/car_tpose_24.pt
  num_keypoints: 24
  num_views: 2
```

### `MVG_SF_V1_Shelf_push/RAFT3D/raft3d/raft3d.py`

This is the model file used for VKITTI training. Keep this path, not the
original human-only `raft3d_original.py`, for VKITTI runs.

Important adaptation points:

- Accepts two VKITTI camera views.
- Uses RGB, depth, intrinsics, sparse keypoint metadata, and DQ/MVGFormer path.
- Keeps dense RAFT3D-style transform/flow prediction.
- The model predicts 2D/3D flow at keypoint-supervised locations.
- It does not predict car keypoints themselves.

Do not casually replace this with the original file. The original is useful for
comparison, but the VKITTI loader/model call expects the adapted interface.

### `MVG_SF_V1_Shelf_push/RAFT3D/scripts/train.py`

This is the training entrypoint used for the W&B runs.

Added/used behavior:

- `fetch_dataloader(...)` can load `vkitti_keypoints_raft`.
- Debug fetch mode verifies dataset tensors before training.
- Validation loop supports VKITTI sparse metrics.
- Checkpoint evaluation mode compares saved checkpoints on validation.
- Visualization writes sparse 2D/3D flow inspection artifacts.
- W&B is optional and controlled by environment/login, not hardcoded tokens.

Metrics:

- `EPE 2D`: endpoint error in pixels.
- `EPE 3D`: endpoint error in meters.
- `1px`: fraction of valid sparse points with 2D error under 1 pixel.
- `5cm`, `10cm`, etc.: fraction of valid sparse points below that 3D error.

## Reimplementation Steps

Clone upstream:

```bash
git clone https://github.com/JagritiSahi/Multi-Camera-feature-for-Robust-SF.git
cd Multi-Camera-feature-for-Robust-SF/MVG_SF_V1_Shelf_push
```

Prepare the environment per upstream, then activate it:

```bash
source ~/.bashrc
conda activate mvgformer_
```

Point the config or environment to VKITTI:

```bash
export VKITTI_ROOT=/path/to/Vkitti
export VKITTI_KEYPOINT_MANIFEST=/path/to/Vkitti/vkitti_car24_world_all_scenes.jsonl
```

Smoke-test the loader:

```bash
python RAFT3D/scripts/train.py \
  --cfg configs/vkitti/vkitti_keypoints_raft.yaml \
  --batch_size 1 \
  --num_workers 0 \
  --gpus 0 \
  --debug_fetch_only
```

Small training run:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python RAFT3D/scripts/train.py \
  --cfg configs/vkitti/vkitti_keypoints_raft.yaml \
  --num_steps 1000 \
  --batch_size 1 \
  --num_workers 0 \
  --gpus 0 \
  --val_interval 200 \
  --viz_interval 500
```

Evaluate checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python RAFT3D/scripts/train.py \
  --cfg configs/vkitti/vkitti_keypoints_raft.yaml \
  --batch_size 1 \
  --num_workers 0 \
  --gpus 0 \
  --eval_checkpoints_dir checkpoints/<checkpoint_dir> \
  --eval_output_dir /path/to/test_results \
  --val_batches 0 \
  --eval_3d_viz 6
```

## Important Caveats

- Annotation generation is upstream of training. It used OpenPifPaf, VKITTI
  depth, `pose.txt`, camera matrices, and cross-variation pose fill.
- The model trains on flow supervision at keypoint locations. It does not learn
  to detect keypoints.
- Confidence comes from the annotation file. Missing keypoints must have
  confidence `0`.
- Official VKITTI forward scene flow should be used for GT 3D flow.
- Do not store W&B or Hugging Face tokens in this repo.

## Other Modified Files Observed

These files were also observed as modified/untracked during work. I did not
document them as core files because the requested core scope was dataset,
`raft3d.py`, and `train.py`.

```text
MVG_SF_V1_Shelf_push/RAFT3D/scripts/eval_vkitti_keypoints.py
MVG_SF_V1_Shelf_push/lib/dataset/panoptic.py
MVG_SF_V1_Shelf_push/lib/models/dq_transformer.py
MVG_SF_V1_Shelf_push/tools/build_vkitti_frame_manifest.py
MVG_SF_V1_Shelf_push/tools/build_vkitti_filtered_manifest.py
MVG_SF_V1_Shelf_push/RAFT3D/experimental/*
```

Ask before treating these as part of the permanent patch set.
