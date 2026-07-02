import torch
import torch.nn as nn
import torch.nn.functional as F


def _debug(*args, **kwargs):
    if os.environ.get("VKITTI_RAFT_SHAPE_DEBUG") == "1":
        print(*args, **kwargs)


def _shape_debug(label, value):
    if os.environ.get("VKITTI_RAFT_SHAPE_DEBUG") == "1":
        if hasattr(value, "shape"):
            print(f"[VKITTI-RAFT-SHAPE] {label}: shape={tuple(value.shape)}")
        else:
            print(f"[VKITTI-RAFT-SHAPE] {label}: {value}")


# lietorch for tangent space backpropogation
from lietorch import SE3

from .blocks.extractor import BasicEncoder
from .blocks.resnet import FPN
from .blocks.corr import CorrBlock
from .blocks.gru import ConvGRU
from .sampler_ops import bilinear_sampler, depth_sampler

from . import projective_ops as pops
from . import se3_field

#================================================================================================
# DQ-RAFT3D Model
#================================================================================================
import sys
import os

# Add root to system path so you can access lib.models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import lib.models.dq_transformer as dq_transformer
from lib.models.dq_transformer import get_mvp
from lib.core.config import config, update_config, update_config_dynamic_input


def init_dq_config(cfg_path='/mnt/MIG_archive24/Datasets/iota/JagritiDatasets/MVG_SF_V1_Shelf/configs/shelf_campus/shelf_knn5-lr4-q1024.yaml', unknown_args=[]):
    update_config(cfg_path)
    update_config_dynamic_input(unknown_args)

#================================================================================================


GRAD_CLIP = .01

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.gru = ConvGRU(hidden_dim)

        self.corr_enc = nn.Sequential(
            nn.Conv2d(196, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*128, 1, padding=0))

        self.flow_enc = nn.Sequential(
            nn.Conv2d(9, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*128, 1, padding=0))

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            nn.Sigmoid(),
            GradientClip())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0),
            GradientClip())


    def forward(self, net, inp, corr, flow, twist, dz, upsample=True):
        motion_info = torch.cat([flow, 10*dz, 10*twist], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0,3,1,2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        net = self.gru(net, inp, cor, mot)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        return net, mask, ae, delta, weight


class RAFT3D(nn.Module):
    def __init__(self, args):

        init_dq_config(getattr(args, 'cfg', None) or 'configs/vkitti/vkitti_keypoints_raft.yaml')

        super(RAFT3D, self).__init__()

        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.dq_model = dq_transformer.get_mvp(config, is_train=False) 
        self.cnet = FPN(output_dim=hdim+3*hdim)
        self.update_block = BasicUpdateBlock(args, hidden_dim=hdim)
        
        # Channel adjustment layer to convert DQ features (64 channels) to RAFT format (128 channels)
        self.channel_adjust = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        
        self.dq_model.eval()
        self.dq_model.to('cuda')

    def create_depth_map_from_reference_points_multiview(self, reference_points, intrinsics, image_height, image_width, num_views=5):
        """
        Convert sparse 3D reference points to dense depth maps for multiple camera views
        
        Args:
            reference_points: [B, N*joints, 3] - 3D points in world/camera coordinates
            intrinsics: [num_views, B, 3, 3] or [num_views, B, 4] - camera intrinsics for each view
            image_height, image_width: target depth map dimensions
            num_views: number of camera views (default 5)
        
        Returns:
            depth_maps: [num_views*B, H, W] - dense depth maps for all views
        """
        batch_size = intrinsics.shape[1]  # Get batch size from intrinsics, not reference_points
        device = reference_points.device
        
        _debug(f"Debug MultiView - reference_points shape: {reference_points.shape}")
        _debug(f"Debug MultiView - intrinsics shape: {intrinsics.shape}")
        _debug(f"Debug MultiView - num_views: {num_views}")
        _debug(f"Debug MultiView - batch_size from intrinsics: {batch_size}")


        # Initialize depth maps for all views
        depth_maps = torch.zeros(num_views * batch_size, image_height, image_width, device=device)
        
        # Handle intrinsics shape: [num_views, batch_size, 3, 3] or [num_views, batch_size, 4]
        if intrinsics.dim() == 4:  # [num_views, batch_size, 3, 3]
            intrinsics_reshaped = intrinsics.view(num_views * batch_size, 3, 3)
        elif intrinsics.dim() == 3:  # [num_views, batch_size, 4] 
            intrinsics_reshaped = intrinsics.view(num_views * batch_size, -1)
        else:
            raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
        
        # Process each view-batch combination
        view_batch_idx = 0
        for view in range(num_views):
            for batch in range(batch_size):
                # For reference points, we need to handle the case where reference_points
                # might have different batch dimension than intrinsics batch dimension
                if reference_points.shape[0] > batch:
                    points_3d = reference_points[batch]  # [N*joints, 3]
                else:
                    # Use the first (or only) set of reference points for all batches
                    points_3d = reference_points[0]  # [N*joints, 3]
                
                # Get intrinsics for this specific view-batch combination
                if intrinsics.dim() == 4:  # [num_views, batch_size, 3, 3]
                    intrinsic_matrix = intrinsics[view, batch]  # [3, 3]
                    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                else:  # [num_views, batch_size, 4+]
                    intrinsic = intrinsics[view, batch]  # [4] or more
                    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
                
                # Project 3D points to 2D image coordinates for this camera view
                X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
                
                # Skip points with zero or negative depth
                valid_depth_mask = Z > 0.1  # minimum depth threshold
                if not valid_depth_mask.any():
                    view_batch_idx += 1
                    continue
                    
                X, Y, Z = X[valid_depth_mask], Y[valid_depth_mask], Z[valid_depth_mask]
                
                # Project to pixel coordinates for this camera view
                u = (fx * X / Z + cx).round().long()
                v = (fy * Y / Z + cy).round().long()
                
                # Filter valid projections
                valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
                
                if valid_mask.any():
                    valid_u = u[valid_mask]
                    valid_v = v[valid_mask]
                    valid_z = Z[valid_mask]
                    
                    # Fill depth map at projected locations
                    for i in range(len(valid_u)):
                        curr_u, curr_v, curr_z = valid_u[i], valid_v[i], valid_z[i]
                        # Take minimum depth if multiple points project to same pixel
                        if depth_maps[view_batch_idx, curr_v, curr_u] == 0:
                            depth_maps[view_batch_idx, curr_v, curr_u] = curr_z
                        else:
                            depth_maps[view_batch_idx, curr_v, curr_u] = min(depth_maps[view_batch_idx, curr_v, curr_u], curr_z)
                
                view_batch_idx += 1
        
        # Fill gaps using interpolation
        depth_maps = self.interpolate_depth_gaps(depth_maps)
        
        return depth_maps.float()

    def _vkitti_depth_hw(self, depth):
        """Return VKITTI depth as [view_batch, H, W] for projective ops."""
        if depth.dim() == 4 and depth.shape[1] == 1:
            return depth[:, 0].float()
        if depth.dim() == 3:
            return depth.float()
        raise ValueError(f"Unexpected VKITTI depth shape: {depth.shape}")



    def interpolate_depth_gaps(self, depth_map):
        """Fill gaps in sparse depth map using interpolation"""
        # Simple approach: use nearest neighbor for empty pixels
        
        for b in range(depth_map.shape[0]):
            depth_slice = depth_map[b]
            mask = depth_slice > 0
            
            if mask.sum() > 4:  # Need at least 4 points for interpolation
                # Get coordinates of valid depth values
                valid_coords = torch.nonzero(mask, as_tuple=False).float().to(depth_map.device)
                valid_depths = depth_slice[mask]
                
                # Create grid for empty pixels only
                empty_mask = depth_slice == 0
                if empty_mask.sum() > 0:
                    empty_coords = torch.nonzero(empty_mask, as_tuple=False).float().to(depth_map.device)
                    
                    # Simple nearest neighbor interpolation for empty pixels
                    if len(valid_coords) > 0 and len(empty_coords) > 0:
                        # Compute distances between empty pixels and valid pixels
                        distances = torch.cdist(empty_coords, valid_coords)
                        nearest_indices = distances.argmin(dim=1)
                        interpolated_depths = valid_depths[nearest_indices]
                        
                        # Fill empty pixels
                        empty_v = empty_coords[:, 0].long()
                        empty_u = empty_coords[:, 1].long()
                        depth_map[b, empty_v, empty_u] = interpolated_depths
            elif mask.sum() > 0:
                # If very few valid points, just fill with mean depth
                mean_depth = depth_slice[mask].mean()
                depth_map[b] = torch.where(depth_map[b] == 0, mean_depth, depth_map[b])
        
        return depth_map.float()

    # def initializer(self, image):
    #     """ Initialize coords and transformation maps """
    #     # image1 = image[0][0]

    #     batch_size, ch, ht, wd = image1.shape
    #     device = image1.device

    #     y0, x0 = torch.meshgrid(torch.arange(ht//8), torch.arange(wd//8))
    #     coords0 = torch.stack([x0, y0], dim=-1).float()
    #     coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

    #     Ts = SE3.Identity(batch_size, ht//8, wd//8, device=device)
    #     return Ts, coords0
    def initializer(self, image1, aggregated=True, batch_size_override=None):
        """ Initialize coords and transformation maps 
        
        Args:
            image1: Raw multi-view images [num_views, C, H, W]
            aggregated: If True, we're using aggregated features
            batch_size_override: Override batch size (use actual batch from aggregated features)
        """

        batch_size, ch, ht, wd = image1.shape
        device = image1.device

        if aggregated and batch_size_override is not None:
            # Use the actual batch size from aggregated features
            # This handles cases where batch_size > 1 (e.g., multiple samples or timesteps)
            effective_batch_size = batch_size_override
            _debug(f"✓ Aggregated mode: Using batch_size={effective_batch_size} from aggregated features")
        elif aggregated:
            # Fallback: assume batch_size=1
            effective_batch_size = 1
            _debug(f"✓ Aggregated mode: Using batch_size=1 for unified scene flow")
        else:
            # Multi-view mode: separate scene flow per view
            effective_batch_size = batch_size
            _debug(f"✓ Multi-view mode: Using batch_size={batch_size} for per-view flows")

        # RAFT3D SE3 state is defined on the 1/8 image grid.
        depth_ht, depth_wd = ht // 8, wd // 8
        
        y0, x0 = torch.meshgrid(torch.arange(depth_ht), torch.arange(depth_wd))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(effective_batch_size, 1, 1, 1).to(device)

        Ts = SE3.Identity(effective_batch_size, depth_ht, depth_wd, device=device)
        
        _debug(f"DEBUG initializer: image1.shape = {image1.shape}")
        _debug(f"DEBUG initializer: effective_batch_size = {effective_batch_size}")
        _debug(f"DEBUG initializer: Ts.data.shape = {Ts.data.shape}")
        _debug(f"DEBUG initializer: coords0.shape = {coords0.shape}")
        
        return Ts, coords0
        
    # def features_and_correlation(self, image , meta):
    # def features_and_correlation(self, image1, image2 , meta=None):
    def features_and_correlation(self, image1, image2, meta=None, image_for_dq=None):
        """
        Args:
        image1, image2: flattened images [10, 3, H, W] for RAFT3D processing
        meta: metadata
        image_for_dq: [image1_t0_list, image2_t1_list] where each list contains 5 tensors
                      of shape [1, 3, H, W] or [3, H, W]
        """
        _debug("(10)model input===>" , image1.shape , image2.shape)
        
        # Get DQ/MVGFormer feature maps for both timestamps. DQ view maps are
        # stereo views at one time, so RAFT temporal correlation must use
        # DQ(t) vs DQ(t+1), not Camera_0(t) vs Camera_1(t).
        if image_for_dq is None:
            raise RuntimeError("image_for_dq is required; refusing BasicEncoder fallback for VKITTI")
        num_views = len(image_for_dq)
        if image1.shape[0] % num_views != 0:
            raise RuntimeError(f"Cannot split flattened image batch {image1.shape[0]} into {num_views} DQ views")
        batch_per_view = image1.shape[0] // num_views
        image2_by_view = image2.view(num_views, batch_per_view, *image2.shape[1:])
        image2_for_dq = [image2_by_view[i] for i in range(num_views)]

        dq_output_t0 = self.dq_model(image_for_dq, meta)
        dq_output_t1 = self.dq_model(image2_for_dq, meta)
        out_t0 = dq_output_t0 if not isinstance(dq_output_t0, tuple) else dq_output_t0[0]
        out_t1 = dq_output_t1 if not isinstance(dq_output_t1, tuple) else dq_output_t1[0]

        def _flatten_dq_views(out, label):
            views = out.get('attn_feature_views') if isinstance(out, dict) else None
            if views is None:
                views = [out.get(f'attn_feature_views_{i}') for i in range(num_views)] if isinstance(out, dict) else []
            if len(views) < num_views or any(v is None or v.dim() != 4 for v in views[:num_views]):
                keys = sorted(list(out.keys())) if isinstance(out, dict) else type(out).__name__
                raise RuntimeError(
                    f"DQ feature maps missing for {label}; refusing BasicEncoder fallback. "
                    f"DQ output keys={keys}."
                )
            fmap = torch.cat([v.float() for v in views[:num_views]], dim=0)
            if fmap.shape[0] != image1.shape[0]:
                raise RuntimeError(f"DQ {label} feature batch {fmap.shape[0]} does not match flattened image batch {image1.shape[0]}")
            return fmap

        fmap1 = self.channel_adjust(_flatten_dq_views(out_t0, 't0'))
        fmap2 = self.channel_adjust(_flatten_dq_views(out_t1, 't1'))
        _shape_debug("fmap1_raw_dq", fmap1)
        _shape_debug("fmap2_raw_dq", fmap2)

        raft_feature_size = (image1.shape[-2] // 8, image1.shape[-1] // 8)
        if fmap1.shape[-2:] != raft_feature_size:
            fmap1 = F.interpolate(fmap1, size=raft_feature_size, mode='bilinear', align_corners=False)
            fmap2 = F.interpolate(fmap2, size=raft_feature_size, mode='bilinear', align_corners=False)
        _shape_debug("fmap1_aligned_raft_1_8", fmap1)
        _shape_debug("fmap2_aligned_raft_1_8", fmap2)
        _debug(f"✓ DQ temporal feature maps: fmap1={fmap1.shape}, fmap2={fmap2.shape}")
        
        _debug(f"✓ Channel-adjusted DQ feature shapes: fmap1={fmap1.shape}, fmap2={fmap2.shape}")
        
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # Use all flattened stereo views so RAFT3D state stays view-aligned.
        context_image = image1
            
        _debug(f"DEBUG: context_image.shape = {context_image.shape}")
        net_inp = self.cnet(context_image)
        _debug(f"DEBUG: net_inp.shape = {net_inp.shape}")
        net, inp = net_inp.split([128, 128*3], dim=1)
        _debug(f"DEBUG: net.shape = {net.shape}, inp.shape = {inp.shape}")

        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        feature_size = fmap1.shape[-2:]
        if inp.shape[-2:] != feature_size:
            inp = F.interpolate(inp, size=feature_size, mode='bilinear', align_corners=True)
            _debug(f"DEBUG: resized inp.shape = {inp.shape}")
        
        if net.shape[-2:] != feature_size:
            net = F.interpolate(net, size=feature_size, mode='bilinear', align_corners=True)
            _debug(f"DEBUG: resized net.shape = {net.shape}")

        return corr_fn, net, inp

    def forward(self, image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=False, meta=None, image_for_dq=None):
    # def forward(self, image , meta ,depth1, depth2, intrinsics, iters=12, train_mode=False):
    # def forward(self, image , meta, intrinsics, iters=12, train_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1 = image[0][0]
        # image2 = image[1][0]
        _debug("(8)model input===>" , image1.shape , image2.shape)
        _shape_debug("image1", image1)
        _shape_debug("image2", image2)
        _shape_debug("depth1_input", depth1)
        _shape_debug("depth2_input", depth2)
        
        # RAFT state is view-flattened: [num_views * batch, ...]. DQ is called
        # inside features_and_correlation for both t and t+1.
        actual_batch_size = image1.shape[0]
        _debug(f"✓ RAFT flattened batch size: {actual_batch_size}")
        
        # Use aggregated mode with actual batch size
        Ts, coords0 = self.initializer(image1, aggregated=True, batch_size_override=actual_batch_size)
        
        _debug("(9)model input===>" , image1.shape , image2.shape)
        # print("meta0===>", len(meta0) , meta0)
        # print("meta00===>" , meta0[0]['center'].shape)
        # print("meta10===>" , meta1[0]['center'].shape)
        corr_fn, net, inp = self.features_and_correlation(image1, image2, meta=meta, image_for_dq=image_for_dq)
        feature_ht, feature_wd = corr_fn.corr_pyramid[0].shape[1:3]
        _shape_debug("feature_ht_feature_wd", (feature_ht, feature_wd))

        # intrinsics and depth at 1/8 resolution - Multi-view handling
        original_intrinsics = intrinsics  # Keep original for multi-view depth creation
        
        _debug(f"Debug - Input intrinsics shape: {intrinsics.shape}")
        _debug(f"Debug - Input image1 shape: {image1.shape}")
        
        # Get DQ transformer output first
        dq_output = self.dq_model(image_for_dq, meta) 
        out = dq_output if not isinstance(dq_output, tuple) else dq_output[0]

        # Extract reference points and aggregated features
        if 'reference_points0' not in out and 'pred_poses' in out:
            out['reference_points0'] = out['pred_poses']['outputs_coord']
        if 'reference_points1' not in out and 'pred_poses' in out:
            out['reference_points1'] = out['pred_poses']['outputs_coord']
        ref_points_0 = out['reference_points0'].float()  # [B, N*joints, 3] - ensure float32
        ref_points_1 = out['reference_points1'].float()  # [B, N*joints, 3] - ensure float32
        actual_batch_size = ref_points_0.shape[0]
        
        _debug(f"Debug - Reference points shape: {ref_points_0.shape}")
        _debug(f"Debug - Aggregated feature batch size: {actual_batch_size}")
        
        # Handle multi-view depth map creation
        _, _, H, W = image1.shape
        stereo_debug = os.environ.get("VKITTI_STEREO_DEBUG") == "1"
        if stereo_debug:
            print("[VKITTI-STEREO-DEBUG][raft3d] before depth-map creation")
            print(f"  image1={tuple(image1.shape)} intrinsics_arg={tuple(intrinsics.shape)}")
            print(f"  image_for_dq_len={len(image_for_dq) if image_for_dq is not None else None} image_for_dq_shapes={[tuple(x.shape) for x in image_for_dq] if image_for_dq is not None else None}")
            print(f"  meta_len={len(meta) if meta is not None else None} meta_view_ids={[m.get('view_id') for m in meta] if meta is not None else None}")
        
        # VKITTI path: use dataset depth maps for RAFT3D geometry.
        # DQ reference points stay feature-side only; they are not used as pseudo-depth.
        if intrinsics.dim() == 5:  # [num_views, batch, timesteps, 3, 3]
            num_views, batch_size_intrinsics, num_timesteps = intrinsics.shape[:3]
            _debug(f"Debug - Multi-view with timesteps: {num_views} views, {batch_size_intrinsics} batches, {num_timesteps} timesteps")
            intrinsics_3x3 = intrinsics[:, :, 0, :, :].contiguous().view(-1, 3, 3)
            depth1 = self._vkitti_depth_hw(depth1)
            depth2 = self._vkitti_depth_hw(depth2)
        elif intrinsics.dim() == 4:  # Multi-view case: [num_views, batch, 3, 3]
            num_views, batch_size_intrinsics = intrinsics.shape[0], intrinsics.shape[1]
            _debug(f"Debug - Multi-view: {num_views} views, {batch_size_intrinsics} batches")
            intrinsics_3x3 = intrinsics.contiguous().view(-1, 3, 3)
            depth1 = self._vkitti_depth_hw(depth1)
            depth2 = self._vkitti_depth_hw(depth2)
        elif intrinsics.dim() == 3:  # [batch, 3, 3] or [num_views, 3, 3]
            intrinsics_3x3 = intrinsics
            depth1 = self._vkitti_depth_hw(depth1)
            depth2 = self._vkitti_depth_hw(depth2)
        else:
            raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")

        expected_depth_batch = intrinsics_3x3.shape[0]
        if depth1.shape[0] != expected_depth_batch or depth2.shape[0] != expected_depth_batch:
            raise ValueError(
                f"VKITTI stereo batch mismatch: depth1={depth1.shape}, "
                f"depth2={depth2.shape}, intrinsics={intrinsics_3x3.shape}"
            )
        if stereo_debug:
            print("[VKITTI-STEREO-DEBUG][raft3d] using VKITTI depth, no camera collapse")
            print(f"  num_views={num_views if 'num_views' in locals() else 'unknown'} batch={batch_size_intrinsics if 'batch_size_intrinsics' in locals() else 'unknown'}")
            print(f"  depth1_hw={tuple(depth1.shape)} depth2_hw={tuple(depth2.shape)} intrinsics_3x3={tuple(intrinsics_3x3.shape)}")


        # Extract intrinsic parameters for RAFT3D
        # After aggregation: intrinsics_3x3 is [batch, 3, 3]
        fx = intrinsics_3x3[:, 0, 0]  # [batch]
        fy = intrinsics_3x3[:, 1, 1]  # [batch]
        cx = intrinsics_3x3[:, 0, 2]  # [batch]
        cy = intrinsics_3x3[:, 1, 2]  # [batch]
        intrinsics_for_raft = torch.stack([fx, fy, cx, cy], dim=1)  # [batch, 4]
        
        _shape_debug("depth1_hw", depth1)
        _shape_debug("depth2_hw", depth2)
        _debug(f"✓ Created depth maps: depth1 {depth1.shape}, depth2 {depth2.shape}")
        _debug(f"✓ RAFT intrinsics: {intrinsics_for_raft.shape}")
        _debug(f"✓ Batch size consistency: Ts={actual_batch_size}, depth={depth1.shape[0]}, intrinsics={intrinsics_for_raft.shape[0]}")
        
        # CRITICAL FIX: Ensure Ts batch size matches depth map batch size
        # This prevents broadcasting errors in lietorch when Ts * X0 is computed
        depth_batch_size = depth1.shape[0]
        if Ts.shape[0] != depth_batch_size:
            _debug(f"⚠️  Batch size mismatch detected! Reinitializing Ts:")
            _debug(f"   - Ts batch size: {Ts.shape[0]}")
            _debug(f"   - Depth batch size: {depth_batch_size}")
            
            # Reinitialize Ts with correct batch size
            device = Ts.device
            depth_ht, depth_wd = 64, 64
            Ts = SE3.Identity(depth_batch_size, depth_ht, depth_wd, device=device)
            
            # Also update coords0 to match
            y0, x0 = torch.meshgrid(
                torch.arange(depth_ht, device=device), 
                torch.arange(depth_wd, device=device))
            coords0 = torch.stack([x0, y0], dim=-1).float()
            coords0 = coords0[None].repeat(depth_batch_size, 1, 1, 1)
            
            _debug(f"✓ Reinitialized Ts with batch_size={depth_batch_size}")
        
        _debug(f"✓ Final batch size consistency: Ts={Ts.shape[0]}, depth={depth1.shape[0]}, intrinsics={intrinsics_for_raft.shape[0]}")
        _debug(f"✓ All tensors ready for aggregated scene flow computation")
        
        # Match original RAFT3D: geometry, correlation, and SE3 all live on the 1/8 grid.
        h_step = w_step = 8
        intrinsics_r8 = intrinsics_for_raft / 8.0
        depth1_r8 = depth1[:, 3::8, 3::8].float()
        depth2_r8 = depth2[:, 3::8, 3::8].float()
        _shape_debug("h_step_w_step", (h_step, w_step))
        _shape_debug("depth1_r8", depth1_r8)
        _shape_debug("depth2_r8", depth2_r8)
        
        # CRITICAL FIX: Reinitialize Ts to match actual downsampled depth spatial dimensions
        # This prevents broadcasting errors when Ts * X0 is computed
        actual_depth_ht, actual_depth_wd = depth1_r8.shape[1], depth1_r8.shape[2]
        if Ts.shape[1] != actual_depth_ht or Ts.shape[2] != actual_depth_wd:
            _debug(f"⚠️  Spatial dimension mismatch detected! Reinitializing Ts:")
            _debug(f"   - Ts spatial dims: {Ts.shape[1]}x{Ts.shape[2]}")
            _debug(f"   - Depth spatial dims: {actual_depth_ht}x{actual_depth_wd}")
            
            # Reinitialize Ts with correct spatial dimensions
            device = Ts.device
            depth_batch_size = depth1_r8.shape[0]
            Ts = SE3.Identity(depth_batch_size, actual_depth_ht, actual_depth_wd, device=device)
            
            # Also update coords0 to match
            y0, x0 = torch.meshgrid(
                torch.arange(actual_depth_ht, device=device), 
                torch.arange(actual_depth_wd, device=device))
            coords0 = torch.stack([x0, y0], dim=-1).float()
            coords0 = coords0[None].repeat(depth_batch_size, 1, 1, 1)
            
            _debug(f"✓ Reinitialized Ts with spatial dims {actual_depth_ht}x{actual_depth_wd}, batch_size={depth_batch_size}")
        
        _debug(f"✓ Final dimensions: Ts={Ts.shape}, depth1_r8={depth1_r8.shape}, depth2_r8={depth2_r8.shape}")
        

        flow_est_list = []
        flow_rev_list = []

        for itr in range(iters):
            Ts = Ts.detach()

            # Debug shape information before projective transform
            _debug(f"DEBUG iteration {itr}: depth1_r8.shape = {depth1_r8.shape}")
            _debug(f"DEBUG iteration {itr}: intrinsics_r8.shape = {intrinsics_r8.shape}")
            _debug(f"DEBUG iteration {itr}: Ts data shape = {Ts.data.shape}")
            
            coords1_xyz, _ = pops.projective_transform(Ts, depth1_r8, intrinsics_r8)
            
            coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1)
            zinv, _ = depth_sampler(1.0/depth2_r8, coords1)

            # Debug: check coordinate and depth dimensions
            _debug(f"Debug coords1 shape: {coords1.shape}")
            _debug(f"Debug depth1_r8 shape: {depth1_r8.shape}")
            coords1_for_corr = coords1.permute(0,3,1,2).contiguous()
            _debug(f"Debug coords1_for_corr shape: {coords1_for_corr.shape}")
            
            # Resize coordinates to match the actual feature dimensions.
            # Correlation function expects coordinates to match feature spatial dimensions
            if coords1_for_corr.shape[2] != feature_ht or coords1_for_corr.shape[3] != feature_wd:
                _debug(f"⚠️  Resizing coordinates from {coords1_for_corr.shape[2]}x{coords1_for_corr.shape[3]} to {feature_ht}x{feature_wd}")
                # Interpolate coordinates to match feature dimensions
                # coords1_for_corr is [B, 2, H, W].
                coords1_for_corr = F.interpolate(
                    coords1_for_corr, 
                    size=(feature_ht, feature_wd), 
                    mode='bilinear', 
                    align_corners=False
                )
                _debug(f"✓ Resized coords1_for_corr to {coords1_for_corr.shape}")
            
            corr = corr_fn(coords1_for_corr)
            _debug(f"DEBUG iteration {itr}: corr shape after corr_fn = {corr.shape}")
            _debug(f"DEBUG iteration {itr}: corr expected channels = 196, got {corr.shape[1]}")
            
            flow = coords1 - coords0

            dz = zinv.unsqueeze(-1) - zinv_proj
            twist = Ts.log()

            # Resize flow and dz to match feature dimensions.
            # Update block expects all inputs to have same spatial dimensions as features
            if flow.shape[1] != feature_ht or flow.shape[2] != feature_wd:
                _debug(f"⚠️  Resizing flow from {flow.shape[1]}x{flow.shape[2]} to {feature_ht}x{feature_wd}")
                # flow is [B, H, W, 2], need to permute to [B, 2, H, W] for interpolation
                flow = flow.permute(0, 3, 1, 2).contiguous()
                flow = F.interpolate(flow, size=(feature_ht, feature_wd), mode='bilinear', align_corners=False)
                flow = flow.permute(0, 2, 3, 1).contiguous()  # Back to [B, H, W, 2]
                _debug(f"✓ Resized flow to {flow.shape}")
            
            if dz.shape[1] != feature_ht or dz.shape[2] != feature_wd:
                _debug(f"⚠️  Resizing dz from {dz.shape[1]}x{dz.shape[2]} to {feature_ht}x{feature_wd}")
                # dz is [B, H, W, 1], need to permute to [B, 1, H, W] for interpolation
                dz = dz.permute(0, 3, 1, 2).contiguous()
                dz = F.interpolate(dz, size=(feature_ht, feature_wd), mode='bilinear', align_corners=False)
                dz = dz.permute(0, 2, 3, 1).contiguous()  # Back to [B, H, W, 1]
                _debug(f"✓ Resized dz to {dz.shape}")
            
            # Also resize twist if needed (twist comes from Ts.log() which has same spatial dims as Ts)
            if twist.shape[1] != feature_ht or twist.shape[2] != feature_wd:
                _debug(f"⚠️  Resizing twist from {twist.shape[1]}x{twist.shape[2]} to {feature_ht}x{feature_wd}")
                # twist is [B, H, W, 6], need to permute to [B, 6, H, W] for interpolation
                twist = twist.permute(0, 3, 1, 2).contiguous()
                twist = F.interpolate(twist, size=(feature_ht, feature_wd), mode='bilinear', align_corners=False)
                twist = twist.permute(0, 2, 3, 1).contiguous()  # Back to [B, H, W, 6]
                _debug(f"✓ Resized twist to {twist.shape}")

            net, mask, ae, delta, weight = \
                self.update_block(net, inp, corr, flow, dz, twist)
            if itr == 0:
                _shape_debug("Ts before upsample/update itr0", Ts.data)
                _shape_debug("mask raw itr0", mask)

            # CRITICAL FIX: Resize mask to match Ts dimensions for upsampling operations
            # mask comes from update_block (64x64), but upsampling expects it to match Ts spatial dims
            Ts_ht, Ts_wd = Ts.shape[1], Ts.shape[2]
            if mask.shape[2] != Ts_ht or mask.shape[3] != Ts_wd:
                _debug(f"⚠️  Resizing mask from {mask.shape[2]}x{mask.shape[3]} to {Ts_ht}x{Ts_wd}")
                # mask is [B, C, H, W], resize to match Ts spatial dims
                mask = F.interpolate(mask, size=(Ts_ht, Ts_wd), mode='bilinear', align_corners=False)
                _debug(f"✓ Resized mask to {mask.shape}")

            # CRITICAL FIX: Resize delta to match coords1_xyz dimensions
            # coords1_xyz has spatial dims from depth1_r8 (e.g., 67x67)
            # delta has spatial dims from update_block (64x64 after resizing inputs)
            coords1_xyz_ht, coords1_xyz_wd = coords1_xyz.shape[1], coords1_xyz.shape[2]
            if delta.shape[2] != coords1_xyz_ht or delta.shape[3] != coords1_xyz_wd:
                _debug(f"⚠️  Resizing delta from {delta.shape[2]}x{delta.shape[3]} to {coords1_xyz_ht}x{coords1_xyz_wd}")
                # delta is [B, C, H, W], resize to match coords1_xyz spatial dims
                delta = F.interpolate(delta, size=(coords1_xyz_ht, coords1_xyz_wd), mode='bilinear', align_corners=False)
                _debug(f"✓ Resized delta to {delta.shape}")

            target = coords1_xyz.permute(0,3,1,2) + delta
            target = target.contiguous()

            # CRITICAL FIX: Resize ae and weight to match target/depth dimensions
            # These are used in se3_field.step_inplace which expects matching spatial dims
            if ae.shape[2] != coords1_xyz_ht or ae.shape[3] != coords1_xyz_wd:
                _debug(f"⚠️  Resizing ae from {ae.shape[2]}x{ae.shape[3]} to {coords1_xyz_ht}x{coords1_xyz_wd}")
                ae = F.interpolate(ae, size=(coords1_xyz_ht, coords1_xyz_wd), mode='bilinear', align_corners=False)
                _debug(f"✓ Resized ae to {ae.shape}")
            
            if weight.shape[2] != coords1_xyz_ht or weight.shape[3] != coords1_xyz_wd:
                _debug(f"⚠️  Resizing weight from {weight.shape[2]}x{weight.shape[3]} to {coords1_xyz_ht}x{coords1_xyz_wd}")
                weight = F.interpolate(weight, size=(coords1_xyz_ht, coords1_xyz_wd), mode='bilinear', align_corners=False)
                _debug(f"✓ Resized weight to {weight.shape}")

            # Gauss-Newton step
            # Ts = se3_field.step(Ts, ae, target, weight, depth1_r8, intrinsics_r8)
            Ts = se3_field.step_inplace(Ts, ae, target, weight, depth1_r8, intrinsics_r8)

            if train_mode:
                flow2d_rev = target.permute(0,2,3,1)[...,:2] - coords0
                flow2d_rev = se3_field.cvx_upsample(8 * flow2d_rev, mask)

                Ts_up = se3_field.upsample_se3(Ts, mask)
                flow2d_est, flow3d_est, valid = pops.induced_flow(Ts_up, depth1, intrinsics)

                flow_est_list.append(flow2d_est)
                flow_rev_list.append(flow2d_rev)

        if train_mode:
            return flow_est_list, flow_rev_list

        # mask is already resized in the loop, so we can use it directly here
        _shape_debug("Ts before final upsample", Ts.data)
        _shape_debug("mask before final upsample", mask)
        Ts_up = se3_field.upsample_se3(Ts, mask)
        _shape_debug("Ts after final upsample", Ts_up.data)
        return Ts_up
