#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CONF_THRESH = 0.1
NUM_KEYPOINTS = 24


def abs_path(root: Path, value):
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def next_frame_path(path: Path, prefix: str, suffix: str):
    try:
        frame = int(path.stem.split("_")[-1]) + 1
    except Exception:
        return None
    return path.with_name(f"{prefix}_{frame:05d}{suffix}")


def scene_flow_path(scene_flow_root: Path, scene: str, variation: str, camera_id: str, frame: int):
    return scene_flow_root / scene / variation / "frames" / "forwardSceneFlow" / f"Camera_{camera_id}" / f"sceneFlow_{frame:05d}.png"


def kp_valid(kp):
    return (
        kp.get("training_keypoint_valid", True)
        and kp.get("x") is not None
        and kp.get("y") is not None
        and kp.get("depth_m") is not None
        and float(kp.get("confidence") or 0.0) > CONF_THRESH
        and kp.get("inside_image", True)
    )


def normalize_24(keypoints):
    out = []
    by_idx = {int(k.get("index", -1)): k for k in keypoints}
    for idx in range(NUM_KEYPOINTS):
        kp = dict(by_idx.get(idx, {}))
        kp.setdefault("index", idx)
        kp.setdefault("x", None)
        kp.setdefault("y", None)
        kp.setdefault("depth_m", None)
        kp.setdefault("confidence", 0.0)
        kp.setdefault("training_keypoint_valid", False)
        out.append(kp)
    return out


def temporal_ok(root: Path, scene_flow_root: Path, sample, counts: Counter):
    scene, variation, frame = sample["scene"], sample["variation"], int(sample["frame"])
    for cam_id, cam in sample.get("cameras", {}).items():
        rgb = abs_path(root, cam.get("rgb"))
        depth = abs_path(root, cam.get("depth"))
        flow = abs_path(root, cam.get("forward_flow"))
        sf = scene_flow_path(scene_flow_root, scene, variation, cam_id, frame)
        checks = [
            ("missing_rgb_t", rgb),
            ("missing_rgb_t1", next_frame_path(rgb, "rgb", ".jpg") if rgb else None),
            ("missing_depth_t", depth),
            ("missing_depth_t1", next_frame_path(depth, "depth", ".png") if depth else None),
            ("missing_forward_flow", flow),
            ("missing_forward_scene_flow", sf),
        ]
        for reason, path in checks:
            if path is None or not path.exists():
                counts[f"frames_skipped_{reason}"] += 1
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/users/multicog/saksham_1/new_proj/vkitti_stereo_car_keypoints_hf/all_scenes_keypoint_level_mask_filter_report/filtered_annotations_keypoint_level_all_scenes.jsonl")
    ap.add_argument("--root", default="/home/users/multicog/saksham_1/new_proj/vkitti_stereo_car_keypoints_hf")
    ap.add_argument("--scene-flow-root", default="/home/users/multicog/saksham_1/new_proj/data/Vkitti/vkitti_2.0.3_forwardSceneFlow")
    ap.add_argument("--output", default="/home/users/multicog/saksham_1/new_proj/vkitti_stereo_car_keypoints_hf/frame_annotations_keypoint_level_all_scenes.jsonl")
    ap.add_argument("--summary", default="/home/users/multicog/saksham_1/new_proj/vkitti_stereo_car_keypoints_hf/frame_annotations_keypoint_level_summary.json")
    args = ap.parse_args()

    root = Path(args.root)
    scene_flow_root = Path(args.scene_flow_root)
    counts = Counter()
    groups = defaultdict(list)

    with open(args.input) as f:
        for line in f:
            sample = json.loads(line)
            counts["track_rows_read"] += 1
            if not temporal_ok(root, scene_flow_root, sample, counts):
                counts["track_rows_skipped_temporal"] += 1
                continue
            groups[(sample["scene"], sample["variation"], int(sample["frame"]))].append(sample)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    per_scene = Counter()
    max_cars = 0
    with open(args.output, "w") as out:
        for (scene, variation, frame), rows in sorted(groups.items()):
            base = rows[0]
            frame_sample = {
                "schema": "vkitti_car_panoptic_frame_v1",
                "scene": scene,
                "variation": variation,
                "frame": frame,
                "cameras": {},
                "cars": [],
            }
            for cam_id, cam in base["cameras"].items():
                sf = scene_flow_path(scene_flow_root, scene, variation, cam_id, frame)
                frame_sample["cameras"][cam_id] = {
                    "rgb": cam["rgb"],
                    "depth": cam["depth"],
                    "forward_flow": cam["forward_flow"],
                    "backward_flow": cam.get("backward_flow"),
                    "forward_scene_flow": str(sf),
                    "intrinsic": cam["intrinsic"],
                    "extrinsic": cam.get("extrinsic"),
                }

            for row in rows:
                common = set()
                valid_by_cam = {}
                for cam_id in ("0", "1"):
                    kps = normalize_24(row["cameras"][cam_id].get("keypoints_2d", []))
                    valid = {int(k["index"]) for k in kps if kp_valid(k)}
                    valid_by_cam[cam_id] = valid
                common = valid_by_cam["0"] & valid_by_cam["1"]
                if not common:
                    counts["cars_skipped_no_stereo_common_keypoints"] += 1
                    continue
                car = {"id": int(row["track_id"]), "cameras": {}}
                for cam_id in ("0", "1"):
                    kps = normalize_24(row["cameras"][cam_id].get("keypoints_2d", []))
                    for kp in kps:
                        idx = int(kp["index"])
                        if idx not in common:
                            kp["training_keypoint_valid"] = False
                            kp["confidence"] = 0.0
                            kp["invalid_reason"] = kp.get("invalid_reason") or "not_stereo_common"
                    car["cameras"][cam_id] = {"keypoints_2d": kps}
                frame_sample["cars"].append(car)
                counts["cars_written"] += 1

            if not frame_sample["cars"]:
                counts["frames_skipped_no_cars"] += 1
                continue
            max_cars = max(max_cars, len(frame_sample["cars"]))
            counts["frames_written"] += 1
            per_scene[scene] += 1
            out.write(json.dumps(frame_sample, separators=(",", ":")) + "\n")

    summary = dict(counts)
    summary["frames_grouped_before_empty_filter"] = len(groups)
    summary["frames_by_scene"] = dict(per_scene)
    summary["max_cars_per_frame"] = max_cars
    summary["output"] = args.output
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
