from __future__ import absolute_import, division, print_function

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


NUM_KEYPOINTS = 24
CONFIDENCE_THRESHOLD = 0.1
SCENE_SPLITS = {
    "train": {"Scene01", "Scene02", "Scene06", "Scene20"},
    "validation": {"Scene18"},
    "val": {"Scene18"},
    "test": {"Scene18"},
}
DEFAULT_SCENE_FLOW_ROOT = Path("/home/users/multicog/saksham_1/new_proj/data/Vkitti/vkitti_2.0.3_forwardSceneFlow")


class VKITTIKeypointsRAFT(Dataset):
    def __init__(self, cfg, subset, is_train, transform=None):
        self.root = Path(cfg.DATASET.ROOT)
        self.vkitti_root = Path(os.environ.get("VKITTI_ROOT", str(self.root)))
        self.transform = transform
        self.num_views = int(cfg.DATASET.CAMERA_NUM)
        self.num_joints = int(cfg.NETWORK.NUM_JOINTS)
        self.maximum_person = int(cfg.MULTI_PERSON.MAX_PEOPLE_NUM)
        self.max_samples = cfg.DATASET.MAX_DATA_NUM
        self.subset = str(subset or "train")
        self.allowed_scenes = SCENE_SPLITS.get(self.subset)
        self.annotation_file = self._choose_manifest()
        self.scene_flow_scale = float(os.environ.get("VKITTI_SCENE_FLOW_SCALE", "1000.0"))
        self.max_source_depth_delta_m = float(os.environ.get("VKITTI_MAX_SOURCE_DEPTH_DELTA_M", "0") or 0)
        self.max_scene_flow_m = float(os.environ.get("VKITTI_MAX_SCENE_FLOW_M", "0") or 0)
        self._table_cache = {}
        self.samples = []

        raw_records = [json.loads(line) for line in open(self.annotation_file)]
        if raw_records and self._is_world_camera_record(raw_records[0]):
            candidates = self._world_records_to_samples(raw_records)
        else:
            candidates = [self._normalize_sample(record) for record in raw_records]

        for sample in candidates:
            if self.allowed_scenes and sample.get("scene") not in self.allowed_scenes:
                continue
            if not all(str(i) in sample.get("cameras", {}) for i in range(self.num_views)):
                continue
            if not sample.get("cars"):
                continue
            if not self._has_required_temporal_files(sample):
                continue
            if not self._has_sparse_supervision(sample):
                continue
            self.samples.append(sample)
            if self.max_samples and len(self.samples) >= int(self.max_samples):
                break

        scene_counts = {}
        car_counts = []
        for sample in self.samples:
            scene_counts[sample.get("scene", "MISSING")] = scene_counts.get(sample.get("scene", "MISSING"), 0) + 1
            car_counts.append(len(sample.get("cars", [])))
        print(
            "Loaded VKITTI frame-level RAFT samples: "
            f"{len(self.samples)} subset={self.subset} scenes={scene_counts} "
            f"max_cars={max(car_counts) if car_counts else 0} "
            f"scene_flow_scale={self.scene_flow_scale} from {self.annotation_file}"
        )
        if self.max_source_depth_delta_m > 0 or self.max_scene_flow_m > 0:
            print(f"VKITTI quality filters: source_depth_delta<={self.max_source_depth_delta_m or 'off'}m, scene_flow<={self.max_scene_flow_m or 'off'}m")

    def _choose_manifest(self):
        explicit = os.environ.get("VKITTI_KEYPOINT_MANIFEST")
        if explicit:
            return Path(explicit)
        world_manifest = self.vkitti_root / "vkitti_car24_world_all_scenes.jsonl"
        if world_manifest.exists():
            return world_manifest
        scene20_manifest = self.vkitti_root / "scene20_car24_world_annotations.jsonl"
        if scene20_manifest.exists():
            return scene20_manifest
        frame_manifest = self.root / "frame_annotations_keypoint_level_all_scenes.jsonl"
        if frame_manifest.exists():
            return frame_manifest
        keypoint_manifest = self.root / "all_scenes_keypoint_level_mask_filter_report" / "filtered_annotations_keypoint_level_all_scenes.jsonl"
        if keypoint_manifest.exists():
            return keypoint_manifest
        preferred = self.root / "filtered_annotations_other_instance.jsonl"
        if preferred.exists():
            return preferred
        return self.root / "annotations.jsonl"

    def _is_world_camera_record(self, record):
        cars = record.get("cars") or []
        return "camera" in record and bool(cars) and "joints24" in cars[0]

    def _world_records_to_samples(self, records):
        grouped = defaultdict(dict)
        for record in records:
            cam_id = int(str(record["camera"]).split("_")[-1])
            key = (record["scene"], record["variation"], int(record["frame"]))
            grouped[key][cam_id] = record

        samples = []
        for (scene, variation, frame), by_camera in sorted(grouped.items()):
            if not all(i in by_camera for i in range(self.num_views)):
                continue
            cameras = {}
            for view_id in range(self.num_views):
                cameras[str(view_id)] = self._camera_paths(scene, variation, frame, view_id)

            car_ids = sorted({
                int(car["id"])
                for record in by_camera.values()
                for car in record.get("cars", [])
            })
            cars = []
            by_cam_car = {
                cam_id: {int(car["id"]): car for car in record.get("cars", [])}
                for cam_id, record in by_camera.items()
            }
            for car_id in car_ids[: self.maximum_person]:
                car = {"id": car_id, "cameras": {}}
                for view_id in range(self.num_views):
                    src = by_cam_car.get(view_id, {}).get(car_id)
                    if src is None:
                        keypoints = self._normalize_keypoints([])
                    else:
                        keypoints = self._project_world_joints(src, cameras[str(view_id)])
                    car["cameras"][str(view_id)] = {"keypoints_2d": keypoints}
                cars.append(car)

            samples.append({
                "schema": "vkitti_car_panoptic_world_frame_v1",
                "scene": scene,
                "variation": variation,
                "frame": frame,
                "cameras": cameras,
                "cars": cars,
            })
        return samples

    def _camera_paths(self, scene, variation, frame, view_id):
        cam = f"Camera_{view_id}"
        intrinsic = self._intrinsic(scene, variation, frame, view_id)
        extrinsic = self._extrinsic(scene, variation, frame, view_id)
        return {
            "rgb": str(Path("vkitti_2.0.3_rgb") / scene / variation / "frames" / "rgb" / cam / f"rgb_{frame:05d}.jpg"),
            "depth": str(Path("vkitti_2.0.3_depth") / scene / variation / "frames" / "depth" / cam / f"depth_{frame:05d}.png"),
            "forward_flow": str(Path("vkitti_2.0.3_forwardFlow") / scene / variation / "frames" / "forwardFlow" / cam / f"flow_{frame:05d}.png"),
            "forward_scene_flow": str(Path("vkitti_2.0.3_forwardSceneFlow") / scene / variation / "frames" / "forwardSceneFlow" / cam / f"sceneFlow_{frame:05d}.png"),
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
        }

    def _read_table_rows(self, scene, variation, name):
        path = self.vkitti_root / "vkitti_2.0.3_textgt" / scene / variation / name
        if path not in self._table_cache:
            rows = []
            with open(path) as f:
                header = f.readline().strip().split()
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        rows.append((header, parts))
            self._table_cache[path] = rows
        return self._table_cache[path]

    def _intrinsic(self, scene, variation, frame, view_id):
        for header, parts in self._read_table_rows(scene, variation, "intrinsic.txt"):
            row = dict(zip(header, parts))
            if int(row["frame"]) == frame and int(row["cameraID"]) == view_id:
                fx, fy = float(row["K[0,0]"]), float(row["K[1,1]"])
                cx, cy = float(row["K[0,2]"]), float(row["K[1,2]"])
                return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        raise FileNotFoundError(f"intrinsic missing: {scene} {variation} frame={frame} camera={view_id}")

    def _extrinsic(self, scene, variation, frame, view_id):
        for _, parts in self._read_table_rows(scene, variation, "extrinsic.txt"):
            if int(parts[0]) == frame and int(parts[1]) == view_id:
                return np.asarray([float(x) for x in parts[2:18]], dtype=np.float32).reshape(4, 4).tolist()
        raise FileNotFoundError(f"extrinsic missing: {scene} {variation} frame={frame} camera={view_id}")

    def _project_world_joints(self, car, cam):
        intr = np.asarray(cam["intrinsic"], dtype=np.float32)
        extr = np.asarray(cam["extrinsic"], dtype=np.float32)
        out = []
        for idx, joint in enumerate(car.get("joints24", [])[:NUM_KEYPOINTS]):
            xw, yw, zw, conf = [float(v) for v in joint]
            kp = {
                "index": idx,
                "x": None,
                "y": None,
                "depth_m": None,
                "confidence": conf,
                "source": (car.get("sources") or ["missing"] * NUM_KEYPOINTS)[idx],
                "training_keypoint_valid": False,
                "inside_image": False,
            }
            if conf > CONFIDENCE_THRESHOLD:
                pc = extr[:3, :3] @ np.array([xw, yw, zw], dtype=np.float32) + extr[:3, 3]
                z = float(pc[2])
                if np.isfinite(z) and z > 0:
                    px = float(intr[0, 0] * pc[0] / z + intr[0, 2])
                    py = float(intr[1, 1] * pc[1] / z + intr[1, 2])
                    inside = 0 <= px < 1242 and 0 <= py < 375
                    kp.update({"x": px, "y": py, "depth_m": z, "training_keypoint_valid": inside, "inside_image": inside})
            out.append(kp)
        return self._normalize_keypoints(out)

    def __len__(self):
        return len(self.samples)

    def _normalize_sample(self, sample):
        if "cars" in sample:
            return sample
        track_id = int(sample["track_id"])
        cameras = {}
        car_cameras = {}
        for cam_id, cam in sample.get("cameras", {}).items():
            cameras[cam_id] = {k: v for k, v in cam.items() if k != "keypoints_2d"}
            if "forward_scene_flow" not in cameras[cam_id]:
                cameras[cam_id]["forward_scene_flow"] = str(self._scene_flow_path(sample["scene"], sample["variation"], cam_id, int(sample["frame"])))
            car_cameras[cam_id] = {"keypoints_2d": self._normalize_keypoints(cam.get("keypoints_2d", []))}
        return {
            "schema": "vkitti_car_panoptic_frame_v1_from_track_row",
            "scene": sample["scene"],
            "variation": sample["variation"],
            "frame": int(sample["frame"]),
            "cameras": cameras,
            "cars": [{"id": track_id, "cameras": car_cameras}],
        }

    def _normalize_keypoints(self, keypoints):
        by_idx = {int(k.get("index", -1)): k for k in keypoints}
        out = []
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

    def _scene_flow_path(self, scene, variation, cam_id, frame):
        root = Path(os.environ.get("VKITTI_FORWARD_SCENE_FLOW_ROOT", str(DEFAULT_SCENE_FLOW_ROOT)))
        return root / scene / variation / "frames" / "forwardSceneFlow" / f"Camera_{cam_id}" / f"sceneFlow_{frame:05d}.png"

    def _abs(self, rel):
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def _next_frame_path(self, rel, prefix, suffix):
        if rel is None:
            return None
        path = self._abs(rel)
        try:
            frame = int(path.stem.split("_")[-1]) + 1
        except ValueError:
            return None
        return str(path.with_name(f"{prefix}_{frame:05d}{suffix}"))

    def _has_required_temporal_files(self, sample):
        for view_id in range(self.num_views):
            cam = sample["cameras"][str(view_id)]
            required = [
                self._abs(cam.get("rgb")),
                self._abs(self._next_frame_path(cam.get("rgb"), "rgb", ".jpg")),
                self._abs(cam.get("depth")),
                self._abs(self._next_frame_path(cam.get("depth"), "depth", ".png")),
                self._abs(cam.get("forward_flow")),
                self._abs(cam.get("forward_scene_flow")),
            ]
            if any(p is None or not Path(p).exists() for p in required):
                return False
        return True

    def _is_visible_keypoint(self, kp):
        if not kp.get("training_keypoint_valid", True):
            return False
        x, y, z = kp.get("x"), kp.get("y"), kp.get("depth_m")
        conf = float(kp.get("confidence") or 0.0)
        if x is None or y is None or z is None or conf <= CONFIDENCE_THRESHOLD:
            return False
        z = float(z)
        return bool(kp.get("inside_image", True) and np.isfinite(z) and z > 0)

    def _has_sparse_supervision(self, sample):
        for view_id in range(self.num_views):
            if not any(
                self._is_visible_keypoint(kp)
                for _, kp in self._iter_view_keypoints(sample, str(view_id))
            ):
                return False
        return True

    def _iter_view_keypoints(self, sample, cam_id):
        for car_index, car in enumerate(sample.get("cars", [])):
            cam = car.get("cameras", {}).get(str(cam_id), {})
            for kp in self._normalize_keypoints(cam.get("keypoints_2d", [])):
                yield car_index, kp

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs, input_t1, meta, meta_t1 = [], [], [], []
        flows, valids, disps, disps_t1, disps_change = [], [], [], [], []
        sceneflows, sceneflow_valids = [], []

        for view_id in range(self.num_views):
            cam = sample["cameras"][str(view_id)]
            image0 = self._read_image(cam["rgb"])
            image1 = self._read_image(self._next_frame_path(cam["rgb"], "rgb", ".jpg"))
            depth0 = self._read_depth(cam.get("depth"), image0.shape[:2])
            depth1 = self._read_depth(self._next_frame_path(cam.get("depth"), "depth", ".png"), image0.shape[:2])
            flow_dense, flow_dense_valid = self._read_flow(cam.get("forward_flow"), image0.shape[:2])
            sf_dense, sf_dense_valid = self._read_scene_flow(cam.get("forward_scene_flow"), image0.shape[:2])
            intr = np.asarray(cam["intrinsic"], dtype=np.float32)
            extr = np.asarray(cam.get("extrinsic", np.eye(4)), dtype=np.float32)
            camera_dict = self._camera_dict(cam, view_id, sample)

            flow_sparse = np.zeros((2, image0.shape[0], image0.shape[1]), dtype=np.float32)
            sceneflow_sparse = np.zeros((3, image0.shape[0], image0.shape[1]), dtype=np.float32)
            valid_sparse = np.zeros((image0.shape[0], image0.shape[1]), dtype=np.float32)
            joints2d, joints2d_vis, joints3d, joints3d_vis, keypoint_conf, track_ids = self._build_keypoints(sample, str(view_id), intr)

            for _, kp in self._iter_view_keypoints(sample, str(view_id)):
                if not self._is_visible_keypoint(kp):
                    continue
                xi, yi = int(round(float(kp["x"]))), int(round(float(kp["y"])))
                if yi < 0 or xi < 0 or yi >= image0.shape[0] or xi >= image0.shape[1]:
                    continue
                if not flow_dense_valid[yi, xi] or not sf_dense_valid[yi, xi]:
                    continue
                u, v = flow_dense[yi, xi]
                z0 = float(kp.get("depth_m") or depth0[yi, xi])
                if not np.isfinite(z0) or z0 <= 0:
                    continue
                source_depth_delta = abs(float(depth0[yi, xi]) - z0)
                if self.max_source_depth_delta_m > 0 and source_depth_delta > self.max_source_depth_delta_m:
                    continue
                scene_flow = sf_dense[yi, xi].astype(np.float32)
                if self.max_scene_flow_m > 0 and float(np.linalg.norm(scene_flow)) > self.max_scene_flow_m:
                    continue
                flow_sparse[:, yi, xi] = [u, v]
                sceneflow_sparse[:, yi, xi] = scene_flow
                valid_sparse[yi, xi] = 1.0

            inputs.append(self._to_tensor(image0))
            input_t1.append(self._to_tensor(image1))
            disp0 = torch.from_numpy(depth0).unsqueeze(0).float()
            disp1 = torch.from_numpy(depth1).unsqueeze(0).float()
            disps.append(disp0)
            disps_t1.append(disp1)
            disps_change.append(disp1 - disp0)
            flows.append(torch.from_numpy(flow_sparse).float())
            valids.append(torch.from_numpy(valid_sparse).float())
            sceneflows.append(torch.from_numpy(sceneflow_sparse).float())
            sceneflow_valids.append(torch.from_numpy(valid_sparse).float())
            meta.append(self._meta(sample, cam, view_id, intr, extr, camera_dict, joints2d, joints2d_vis, joints3d, joints3d_vis, keypoint_conf, track_ids))
            meta_t1.append(self._meta(sample, cam, view_id, intr, extr, camera_dict, joints2d, joints2d_vis, joints3d, joints3d_vis, keypoint_conf, track_ids))

        return inputs, input_t1, meta, meta_t1, flows, valids, disps, disps_t1, disps_change, sceneflows, sceneflow_valids

    def _read_image(self, rel):
        img = cv2.imread(str(self._abs(rel)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self._abs(rel))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_depth(self, rel, shape):
        img = cv2.imread(str(self._abs(rel)), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(self._abs(rel))
        if img.ndim == 3:
            img = img[:, :, 0]
        return img.astype(np.float32) / 100.0

    def _read_flow(self, rel, shape):
        flow = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
        valid = np.zeros(shape, dtype=bool)
        img = cv2.imread(str(self._abs(rel)), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(self._abs(rel))
        if img.ndim == 3 and img.shape[2] >= 3:
            rgb = img[:, :, ::-1].astype(np.float32)
            flow[..., 0] = (rgb[..., 0] - 32768.0) / 64.0
            flow[..., 1] = (rgb[..., 1] - 32768.0) / 64.0
            valid = rgb[..., 2] > 0
        return flow, valid

    def _read_scene_flow(self, rel, shape):
        flow = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
        valid = np.zeros(shape, dtype=bool)
        img = cv2.imread(str(self._abs(rel)), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(self._abs(rel))
        if img.ndim == 3 and img.shape[2] >= 3:
            rgb = img[:, :, ::-1].astype(np.float32)
            flow[..., 0] = (rgb[..., 0] - 32768.0) / self.scene_flow_scale
            flow[..., 1] = (rgb[..., 1] - 32768.0) / self.scene_flow_scale
            flow[..., 2] = (rgb[..., 2] - 32768.0) / self.scene_flow_scale
            valid = np.ones(shape, dtype=bool)
        return flow, valid

    def _to_tensor(self, image):
        if self.transform is not None:
            return self.transform(image)
        return torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

    def _unproject(self, x, y, z, intr):
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        return np.array([(x - cx) * z / fx, (y - cy) * z / fy, z], dtype=np.float32)

    def _camera_dict(self, cam, view_id, sample):
        intr = np.asarray(cam["intrinsic"], dtype=np.float32)
        extr = np.asarray(cam.get("extrinsic", np.eye(4)), dtype=np.float32)
        if int(view_id) == 0:
            r = np.eye(3, dtype=np.float32)
            t = np.zeros((3, 1), dtype=np.float32)
        else:
            cam0 = sample["cameras"]["0"]
            extr0 = np.asarray(cam0.get("extrinsic", np.eye(4)), dtype=np.float32)
            world0_to_cam = np.linalg.inv(extr) @ extr0
            r = world0_to_cam[:3, :3].astype(np.float32)
            trans = world0_to_cam[:3, 3:4].astype(np.float32)
            t = (-r.T @ trans).astype(np.float32)
        return {
            "fx": np.array(intr[0, 0], dtype=np.float32),
            "fy": np.array(intr[1, 1], dtype=np.float32),
            "cx": np.array(intr[0, 2], dtype=np.float32),
            "cy": np.array(intr[1, 2], dtype=np.float32),
            "R": r,
            "T": t,
            "standard_T": t.copy(),
            "k": np.zeros((3, 1), dtype=np.float32),
            "p": np.zeros((2, 1), dtype=np.float32),
        }

    def _build_keypoints(self, sample, cam_id, intr):
        cars = sample.get("cars", [])[: self.maximum_person]
        n = len(cars)
        joints2d = np.zeros((n, NUM_KEYPOINTS, 2), dtype=np.float32)
        joints2d_vis = np.zeros((n, NUM_KEYPOINTS, 2), dtype=np.float32)
        joints3d = np.zeros((n, NUM_KEYPOINTS, 3), dtype=np.float32)
        joints3d_vis = np.zeros((n, NUM_KEYPOINTS, 3), dtype=np.float32)
        keypoint_conf = np.zeros((n, NUM_KEYPOINTS), dtype=np.float32)
        track_ids = np.zeros((n,), dtype=np.int64)
        for car_i, car in enumerate(cars):
            track_ids[car_i] = int(car["id"])
            cam = car.get("cameras", {}).get(str(cam_id), {})
            for kp in self._normalize_keypoints(cam.get("keypoints_2d", [])):
                idx = int(kp.get("index", -1))
                if idx < 0 or idx >= NUM_KEYPOINTS:
                    continue
                conf = float(kp.get("confidence") or 0.0)
                keypoint_conf[car_i, idx] = conf
                if not self._is_visible_keypoint(kp):
                    continue
                x, y, z = float(kp["x"]), float(kp["y"]), float(kp["depth_m"])
                joints2d[car_i, idx] = [x, y]
                joints2d_vis[car_i, idx] = [1.0, 1.0]
                joints3d[car_i, idx] = self._unproject(x, y, z, intr)
                joints3d_vis[car_i, idx] = [1.0, 1.0, 1.0]
        return joints2d, joints2d_vis, joints3d, joints3d_vis, keypoint_conf, track_ids

    def _meta(self, sample, cam, view_id, intr, extr, camera_dict, joints2d, joints2d_vis, joints3d, joints3d_vis, keypoint_conf, track_ids):
        h, w = 375, 1242
        num_person = int(joints2d.shape[0])
        return {
            "image": str(self._abs(cam["rgb"])),
            "scene": sample["scene"],
            "variation": sample["variation"],
            "frame": int(sample["frame"]),
            "track_ids": track_ids,
            "track_id": int(track_ids[0]) if len(track_ids) else -1,
            "view_id": int(view_id),
            "num_person": num_person,
            "joints_3d": joints3d,
            "joints_3d_vis": joints3d_vis,
            "joints_3d_voxelpose_pred": np.zeros((max(self.maximum_person, num_person), NUM_KEYPOINTS, 5), dtype=np.float32),
            "roots_3d": joints3d[:, 0] if num_person else np.zeros((0, 3), dtype=np.float32),
            "joints": joints2d,
            "joints_vis": joints2d_vis,
            "keypoint_confidence": keypoint_conf,
            "confidence_threshold": np.array(CONFIDENCE_THRESHOLD, dtype=np.float32),
            "center": np.array([w / 2.0, h / 2.0], dtype=np.float32),
            "scale": np.array([w / 200.0, h / 200.0], dtype=np.float32),
            "rotation": 0.0,
            "camera": camera_dict,
            "camera_Intri": intr.astype(np.float32),
            "camera_Extri": extr.astype(np.float32),
            "camera_R": extr[:3, :3].astype(np.float32),
            "camera_focal": np.array([intr[0, 0], intr[1, 1], 1.0], dtype=np.float32),
            "camera_T": extr[:3, 3].astype(np.float32),
            "camera_standard_T": extr[:3, 3].astype(np.float32),
            "affine_trans": np.eye(3, dtype=np.float32),
            "inv_affine_trans": np.eye(3, dtype=np.float32),
            "aug_trans": np.eye(3, dtype=np.float32),
        }
