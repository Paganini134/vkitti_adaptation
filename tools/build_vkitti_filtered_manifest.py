
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2


def mask_path(root, scene, variation, camera_id, frame):
    return (Path(root) / scene / variation / "frames" / "instanceSegmentation" /
            f"Camera_{camera_id}" / f"instancegt_{int(frame):05d}.png")


def read_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def point_status(mask, x, y, target_value):
    if x is None or y is None:
        return "missing", None
    xi, yi = int(round(float(x))), int(round(float(y)))
    if mask is None:
        return "mask_missing", None
    h, w = mask.shape[:2]
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return "outside_image", None
    value = int(mask[yi, xi])
    if value == target_value:
        return "target", value
    if value == 0:
        return "background", value
    return "other_instance", value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--instance-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--rejections-csv", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    counts = Counter()
    by_scene_variation = Counter()
    reject_rows = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rejections_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.annotations) as fin, open(args.output, "w") as fout:
        for line in fin:
            if args.limit is not None and counts["frames_before_other_instance_filter"] >= args.limit:
                break
            sample = json.loads(line)
            counts["frames_before_other_instance_filter"] += 1
            scene = sample["scene"]
            variation = sample["variation"]
            frame = sample["frame"]
            track_id = int(sample["track_id"])
            target_value = track_id + 1
            reject = False
            reject_detail = None

            for cam_key in ("0", "1"):
                cam = sample.get("cameras", {}).get(cam_key)
                if cam is None:
                    reject = True
                    reject_detail = (cam_key, "missing_camera", None, None, None)
                    break
                mpath = mask_path(args.instance_root, scene, variation, int(cam_key), frame)
                mask = read_mask(mpath)
                if mask is None:
                    reject = True
                    reject_detail = (cam_key, "missing_mask", None, None, str(mpath))
                    break

                for kp in cam.get("keypoints_2d", []):
                    status, value = point_status(mask, kp.get("x"), kp.get("y"), target_value)
                    kp["instance_mask_status"] = status
                    kp["instance_mask_value"] = value
                    counts[f"keypoint_{status}"] += 1
                    if status == "other_instance":
                        reject = True
                        reject_detail = (cam_key, status, kp.get("index"), value, str(mpath))
                        break
                if reject:
                    break

            if reject:
                counts["frames_rejected_other_instance_overlap"] += 1
                cam_key, reason, kp_index, mask_value, mpath = reject_detail
                by_scene_variation[(scene, variation, cam_key, reason)] += 1
                reject_rows.append({
                    "scene": scene,
                    "variation": variation,
                    "frame": frame,
                    "track_id": track_id,
                    "camera": cam_key,
                    "reason": reason,
                    "keypoint_index": kp_index,
                    "mask_value": mask_value,
                    "mask_path": mpath,
                })
                continue

            counts["frames_after_other_instance_filter"] += 1
            fout.write(json.dumps(sample) + "\n")

    metrics = {
        "counts": dict(counts),
        "rejections_by_scene_variation_camera_reason": [
            {"scene": k[0], "variation": k[1], "camera": k[2], "reason": k[3], "count": v}
            for k, v in sorted(by_scene_variation.items())
        ],
    }
    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.rejections_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "variation", "frame", "track_id", "camera", "reason", "keypoint_index", "mask_value", "mask_path"])
        writer.writeheader()
        writer.writerows(reject_rows)
    print(json.dumps(metrics["counts"], indent=2))


if __name__ == "__main__":
    main()
