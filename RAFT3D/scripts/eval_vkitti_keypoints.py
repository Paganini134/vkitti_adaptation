import sys
from pathlib import Path
sys.path.append('.')
sys.path.append('./lib')
sys.path.append('./RAFT3D/scripts')

import argparse
import csv
import importlib
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lietorch import SE3
import RAFT3D.raft3d.projective_ops as pops
from utils import normalize_image
from lib.core.config import config, update_config
import lib.dataset as dataset

DEPTH_SCALE = 0.05


def flatten_leading(x):
    if x.dim() > 4:
        n = 1
        for d in x.shape[:-3]:
            n *= d
        return x.view(n, *x.shape[-3:])
    return x


def prepare(image1, image2, depth1, depth2):
    image1 = flatten_leading(image1)
    image2 = flatten_leading(image2)
    depth1 = flatten_leading(depth1)
    depth2 = flatten_leading(depth2)
    if depth1.dim() == 3:
        depth1 = depth1.unsqueeze(1)
    if depth2.dim() == 3:
        depth2 = depth2.unsqueeze(1)
    h, w = image1.shape[-2:]
    pad_h = (-h) % 8
    pad_w = (-w) % 8
    image1 = F.pad(image1, [0, pad_w, 0, pad_h], mode='replicate')
    image2 = F.pad(image2, [0, pad_w, 0, pad_h], mode='replicate')
    depth1 = F.pad(depth1, [0, pad_w, 0, pad_h], mode='replicate') * DEPTH_SCALE
    depth2 = F.pad(depth2, [0, pad_w, 0, pad_h], mode='replicate') * DEPTH_SCALE
    return normalize_image(image1.float()), normalize_image(image2.float()), depth1.float(), depth2.float(), (h, w)


def intrinsics_from_meta(meta, meta_t1):
    k0, k1 = [], []
    for m0, m1 in zip(meta, meta_t1):
        a = m0['camera_Intri']
        b = m1['camera_Intri']
        if a.shape[0] == 4:
            a = a[:3, :3]
        if b.shape[0] == 4:
            b = b[:3, :3]
        if a.dim() > 2:
            a = a[0]
        if b.dim() > 2:
            b = b[0]
        k0.append(a)
        k1.append(b)
    return torch.stack([torch.stack(k0, 0), torch.stack(k1, 0)], dim=1).unsqueeze(1).float().cuda()


def run_model(model, blob):
    inputs, input_t1, meta, meta_t1, flows, valids, disps, disps_t1, _, sceneflows, sceneflow_valids = blob
    image1 = torch.stack(inputs, dim=0).cuda()
    image2 = torch.stack(input_t1, dim=0).cuda()
    depth1 = torch.stack(disps, dim=0).cuda()
    depth2 = torch.stack(disps_t1, dim=0).cuda()
    flow_gt = torch.stack(flows, dim=0).cuda()[:, 0].permute(0, 2, 3, 1)
    flow3d_gt = torch.stack(sceneflows, dim=0).cuda()[:, 0].permute(0, 2, 3, 1)
    valid = torch.stack(sceneflow_valids, dim=0).cuda()[:, 0] > 0
    intrinsics = intrinsics_from_meta(meta, meta_t1)

    image1, image2, depth1, depth2, (h, w) = prepare(image1, image2, depth1, depth2)
    num_views = config.DATASET.CAMERA_NUM
    image_for_dq = [image1[i:i + 1] for i in range(num_views)]

    ts = model(image1, image2, depth1, depth2, intrinsics, iters=12, meta=meta, image_for_dq=image_for_dq)
    if ts.shape[0] == 1:
        ts = SE3(ts.data.repeat(num_views, 1, 1, 1))
    intrinsics_for_flow = intrinsics[:, 0, 0]
    depth_for_flow = depth1[:num_views]
    flow_pred, flow3d_pred, _ = pops.induced_flow(ts, depth_for_flow, intrinsics_for_flow)
    flow_pred = flow_pred[:, :h, :w, :2]
    flow3d_pred = flow3d_pred[:, :h, :w] / DEPTH_SCALE
    return meta, flow_pred, flow_gt, flow3d_pred, flow3d_gt, valid


def scalar(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().flatten()[0].item()
    elif isinstance(x, (list, tuple)):
        x = x[0]
    return x


def draw_overlay(meta, flow_pred, flow_gt, valid, out_path, max_points=24):
    views = []
    for v, m in enumerate(meta):
        img_path = m['image'][0] if isinstance(m['image'], list) else m['image']
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ys, xs = torch.where(valid[v])
        for n, (y, x) in enumerate(zip(ys[:max_points], xs[:max_points])):
            xi, yi = int(x.item()), int(y.item())
            pg = flow_gt[v, yi, xi].detach().cpu().numpy()
            pp = flow_pred[v, yi, xi].detach().cpu().numpy()
            cv2.circle(img, (xi, yi), 4, (0, 255, 255), -1)
            cv2.arrowedLine(img, (xi, yi), (int(xi + pg[0]), int(yi + pg[1])), (0, 255, 0), 2, tipLength=0.25)
            cv2.arrowedLine(img, (xi, yi), (int(xi + pp[0]), int(yi + pp[1])), (0, 0, 255), 2, tipLength=0.25)
            cv2.putText(img, str(n), (xi + 5, yi - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        views.append(img)
    if views:
        canvas = np.concatenate(views, axis=1)
        cv2.putText(canvas, 'yellow=keypoint green=GT flow red=pred flow', (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--subset', choices=['validation', 'val', 'test'], required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--visualize', type=int, default=8)
    args = ap.parse_args()

    update_config(args.cfg)
    config.DATASET.TEST_SUBSET = args.subset
    config.DATASET.MAX_DATA_NUM = args.max_samples

    out = Path(args.output_dir) / args.subset
    (out / 'overlays').mkdir(parents=True, exist_ok=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ds = eval('dataset.' + config.DATASET.TEST_DATASET)(config, args.subset, False, transforms.Compose([transforms.ToTensor(), normalize]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    RAFT3D = importlib.import_module('RAFT3D.raft3d.raft3d').RAFT3D
    model = torch.nn.DataParallel(RAFT3D(argparse.Namespace(radius=32)))
    model.cuda().eval()
    model.load_state_dict(torch.load(args.ckpt), strict=False)

    rows = []
    total_valid = 0
    total_empty = 0
    sum_epe2d = 0.0
    sum_epe3d = 0.0
    visualized = 0
    with torch.no_grad():
        for i, blob in enumerate(dl):
            meta, fp, fg, f3p, f3g, valid = run_model(model, blob)
            n = int(valid.sum().item())
            total_valid += n
            if n == 0:
                total_empty += 1
                e2 = e3 = float('nan')
            else:
                e2_all = torch.linalg.norm(fp - fg, dim=-1)[valid]
                e3_all = torch.linalg.norm(f3p - f3g, dim=-1)[valid]
                e2 = float(e2_all.mean().item())
                e3 = float(e3_all.mean().item())
                sum_epe2d += float(e2_all.sum().item())
                sum_epe3d += float(e3_all.sum().item())
            m = meta[0]
            rows.append({
                'sample_index': i,
                'scene': m['scene'][0] if isinstance(m['scene'], list) else m['scene'],
                'variation': m['variation'][0] if isinstance(m['variation'], list) else m['variation'],
                'frame': int(scalar(m['frame'])),
                'track_id': int(scalar(m['track_id'])),
                'valid_keypoint_pixels': n,
                'epe2d_px': e2,
                'epe3d_m': e3,
            })
            if n > 0 and visualized < args.visualize:
                draw_overlay(meta, fp, fg, valid, out / 'overlays' / f'sample_{i:04d}.png')
                visualized += 1

    summary = {
        'subset': args.subset,
        'checkpoint': args.ckpt,
        'samples': len(rows),
        'empty_valid_samples': total_empty,
        'valid_keypoint_pixels': total_valid,
        'mean_epe2d_px_weighted': sum_epe2d / max(total_valid, 1),
        'mean_epe3d_m_weighted': sum_epe3d / max(total_valid, 1),
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(out / 'per_sample_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['sample_index'])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
