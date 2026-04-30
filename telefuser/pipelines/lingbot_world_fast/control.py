from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


def _se3_inverse(poses: torch.Tensor) -> torch.Tensor:
    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3:]
    rot_inv = rot.transpose(-1, -2)
    trans_inv = -torch.bmm(rot_inv, trans)
    out = torch.eye(4, device=poses.device, dtype=poses.dtype)[None].repeat(poses.shape[0], 1, 1)
    out[:, :3, :3] = rot_inv
    out[:, :3, 3:] = trans_inv
    return out


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2c = _se3_inverse(c2ws_mat[0:1])
    relative = torch.matmul(ref_w2c, c2ws_mat)
    relative[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise and relative.shape[0] > 1:
        relative[1:] = torch.bmm(_se3_inverse(relative[:-1]), relative[1:])
    if normalize_trans:
        trans = relative[:, :3, 3]
        max_norm = torch.norm(trans, dim=-1).max()
        if float(max_norm) > 0:
            relative[:, :3, 3] = trans / max_norm
    return relative


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    trans_interp = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    trans = trans_interp(tgt_indices)

    src_rot = Rotation.from_matrix(src_rot_mat)
    quats = src_rot.as_quat().copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_rot = Rotation.from_quat(quats)
    rot = Slerp(src_indices, src_rot)(tgt_indices).as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4), dtype=np.float32)
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = trans
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses)


def get_ks_transformed(
    ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
) -> torch.Tensor:
    fx, fy, cx, cy = ks.chunk(4, dim=-1)
    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    out = torch.zeros_like(ks)
    out[:, 0:1] = fx_resize
    out[:, 1:2] = fy_resize
    out[:, 2:3] = cx_resize - crop_offset_x
    out[:, 3:4] = cy_resize - crop_offset_y
    return out


def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2) + bias
    return grid_xy[None].repeat(n_frames, 1, 1)


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    ks: torch.Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
) -> torch.Tensor:
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    fx, fy, cx, cy = ks.chunk(4, dim=-1)

    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    z = torch.ones_like(i)
    x = (i - cx) / fx * z
    y = (j - cy) / fy * z

    dirs = torch.stack([x, y, z], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    rays_d = dirs @ c2ws_mat[:, :3, :3].transpose(-1, -2)

    if only_rays_d:
        return rays_d.view(n_frames, height, width, 3)

    rays_o = c2ws_mat[:, :3, 3][:, None, :].expand_as(rays_d)
    return torch.cat([rays_o, rays_d], dim=-1).view(n_frames, height, width, 6)


@dataclass
class CameraControlChunk:
    control_tensor: torch.Tensor
    num_latent_frames: int
    control_type: str


def build_camera_control_chunk(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    latent_h: int,
    latent_w: int,
    height: int,
    width: int,
) -> CameraControlChunk:
    plucker = get_plucker_embeddings(poses, intrinsics, height, width, only_rays_d=False)
    control = rearrange(
        plucker,
        "f (h c1) (w c2) c -> 1 (c c1 c2) f h w",
        c1=max(1, height // latent_h),
        c2=max(1, width // latent_w),
        h=latent_h,
        w=latent_w,
    ).contiguous()
    return CameraControlChunk(control_tensor=control, num_latent_frames=control.shape[2], control_type="cam")


def build_action_control_chunk(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    action: torch.Tensor,
    latent_h: int,
    latent_w: int,
    height: int,
    width: int,
) -> CameraControlChunk:
    plucker = get_plucker_embeddings(poses, intrinsics, height, width, only_rays_d=True)
    action_map = action[:, None, None, :].repeat(1, height, width, 1)
    combined = torch.cat([plucker, action_map], dim=-1)
    control = rearrange(
        combined,
        "f (h c1) (w c2) c -> 1 (c c1 c2) f h w",
        c1=max(1, height // latent_h),
        c2=max(1, width // latent_w),
        h=latent_h,
        w=latent_w,
    ).contiguous()
    return CameraControlChunk(control_tensor=control, num_latent_frames=control.shape[2], control_type="act")


def load_camera_control_inputs(action_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(action_path)
    poses = np.load(root / "poses.npy")
    intrinsics = np.load(root / "intrinsics.npy")
    return poses, intrinsics


def load_action_control_inputs(action_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(action_path)
    poses = np.load(root / "poses.npy")
    intrinsics = np.load(root / "intrinsics.npy")
    action = np.load(root / "action.npy")
    return poses, intrinsics, action
