from typing import List, Dict, Callable, Tuple, Optional
import torch
import torch.nn.functional as F
import functools


def get_crop_and_resize_matrix(
        box: torch.Tensor, target_shape: Tuple[int, int],
        target_face_scale: float = 1.0, make_square_crop: bool = True,
        offset_xy: Optional[Tuple[float, float]] = None, align_corners: bool = True,
        offset_box_coords: bool = False) -> torch.Tensor:
    if offset_xy is None:
        offset_xy = (0.0, 0.0)

    x1, y1, x2, y2 = box.split(1, dim=1)  # b x 1
    cx = (x1 + x2) / 2 + offset_xy[0]
    cy = (y1 + y2) / 2 + offset_xy[1]
    rx = (x2 - x1) / 2 / target_face_scale
    ry = (y2 - y1) / 2 / target_face_scale
    if make_square_crop:
        rx = ry = torch.maximum(rx, ry)

    x1, y1, x2, y2 = cx - rx, cy - ry, cx + rx, cy + ry

    h, w, *_ = target_shape

    zeros_pl = torch.zeros_like(x1)
    ones_pl = torch.ones_like(x1)

    if align_corners:
        ax = 1.0 / (x2 - x1) * (w - 1)
        ay = 1.0 / (y2 - y1) * (h - 1)
        matrix = torch.cat([
            ax, zeros_pl, -x1 * ax,
            zeros_pl, ay, -y1 * ay,
            zeros_pl, zeros_pl, ones_pl
        ], dim=1).reshape(-1, 3, 3) 
    else:
        if offset_box_coords:
            x1, x2, y1, y2 = x1-0.5, x2-0.5, y1-0.5, y2-0.5
        ax = 1.0 / (x2 - x1) * w
        ay = 1.0 / (y2 - y1) * h
        matrix = torch.cat([
            ax, zeros_pl, -x1 * ax - 0.5*ones_pl,
            zeros_pl, ay, -y1 * ay - 0.5*ones_pl,
            zeros_pl, zeros_pl, ones_pl
        ], dim=1).reshape(-1, 3, 3) 
    return matrix


def get_similarity_transform_matrix(
        from_pts: torch.Tensor, to_pts: torch.Tensor) -> torch.Tensor:
    mfrom = from_pts.mean(dim=1, keepdim=True)  
    mto = to_pts.mean(dim=1, keepdim=True)  

    a1 = (from_pts - mfrom).square().sum([1, 2], keepdim=False)  
    c1 = ((to_pts - mto) * (from_pts - mfrom)).sum([1, 2], keepdim=False)  

    to_delta = to_pts - mto
    from_delta = from_pts - mfrom
    c2 = (to_delta[:, :, 0] * from_delta[:, :, 1] - to_delta[:,
          :, 1] * from_delta[:, :, 0]).sum([1], keepdim=False)  

    a = c1 / a1
    b = c2 / a1
    dx = mto[:, 0, 0] - a * mfrom[:, 0, 0] - b * mfrom[:, 0, 1]  
    dy = mto[:, 0, 1] + b * mfrom[:, 0, 0] - a * mfrom[:, 0, 1]  

    ones_pl = torch.ones_like(a1)
    zeros_pl = torch.zeros_like(a1)

    return torch.stack([
        a, b, dx,
        -b, a, dy,
        zeros_pl, zeros_pl, ones_pl,
    ], dim=-1).reshape(-1, 3, 3)


@functools.lru_cache()
def _standard_face_pts():
    pts = torch.tensor([
        196.0, 226.0,
        316.0, 226.0,
        256.0, 286.0,
        220.0, 360.4,
        292.0, 360.4], dtype=torch.float32) / 256.0 - 1.0
    return torch.reshape(pts, (5, 2))


def get_face_align_matrix(
        face_pts: torch.Tensor, target_shape: Tuple[int, int],
        target_face_scale: float = 1.0, offset_xy: Optional[Tuple[float, float]] = None,
        target_pts: Optional[torch.Tensor] = None):

    if target_pts is None:
        with torch.no_grad():
            std_pts = _standard_face_pts().to(face_pts)  # [-1 1]
            h, w, *_ = target_shape
            target_pts = (std_pts * target_face_scale + 1) * \
                torch.tensor([w-1, h-1]).to(face_pts) / 2.0
            if offset_xy is not None:
                target_pts[:, 0] += offset_xy[0]
                target_pts[:, 1] += offset_xy[1]
    else:
        target_pts = target_pts.to(face_pts)

    if target_pts.dim() == 2:
        target_pts = target_pts.unsqueeze(0)
    if target_pts.size(0) == 1:
        target_pts = target_pts.broadcast_to(face_pts.shape)

    assert target_pts.shape == face_pts.shape

    return get_similarity_transform_matrix(face_pts, target_pts)


@functools.lru_cache(maxsize=128)
def _meshgrid(h, w) -> Tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(h).float(),
                            torch.arange(w).float())
    return yy, xx


def _forge_grid(batch_size: int, device: torch.device,
                output_shape: Tuple[int, int],
                fn: Callable[[torch.Tensor], torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w, *_ = output_shape
    yy, xx = _meshgrid(h, w) 
    yy = yy.unsqueeze(0).broadcast_to(batch_size, h, w).to(device)
    xx = xx.unsqueeze(0).broadcast_to(batch_size, h, w).to(device)

    in_xxyy = torch.stack(
        [xx, yy], dim=-1).reshape([batch_size, h*w, 2])  
    out_xxyy: torch.Tensor = fn(in_xxyy) 
    return out_xxyy.reshape(batch_size, h, w, 2)


def _safe_arctanh(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    return torch.clamp(x, -1+eps, 1-eps).arctanh()


def inverted_tanh_warp_transform(coords: torch.Tensor, matrix: torch.Tensor,
                                 warp_factor: float, warped_shape: Tuple[int, int]):
  
    h, w, *_ = warped_shape

    w_h = torch.tensor([[w, h]]).to(coords)

    if warp_factor > 0:
        coords = coords / w_h * 2 - 1

        nl_part1 = coords > 1.0 - warp_factor
        nl_part2 = coords < -1.0 + warp_factor

        ret_nl_part1 = _safe_arctanh(
            (coords - 1.0 + warp_factor) /
            warp_factor) * warp_factor + \
            1.0 - warp_factor
        ret_nl_part2 = _safe_arctanh(
            (coords + 1.0 - warp_factor) /
            warp_factor) * warp_factor - \
            1.0 + warp_factor

        coords = torch.where(nl_part1, ret_nl_part1,
                             torch.where(nl_part2, ret_nl_part2, coords))

        coords = (coords + 1) / 2 

    coords_homo = torch.cat(
        [coords, torch.ones_like(coords[:, :, [0]])], dim=-1)  

    inv_matrix = torch.linalg.inv(matrix) 
    coords_homo = torch.bmm(
        coords_homo, inv_matrix.permute(0, 2, 1))  
    return coords_homo[:, :, :2] / coords_homo[:, :, [2, 2]]


def tanh_warp_transform(
        coords: torch.Tensor, matrix: torch.Tensor,
        warp_factor: float, warped_shape: Tuple[int, int]):
    h, w, *_ = warped_shape
   
    w_h = torch.tensor([[w, h]]).to(coords)

    coords_homo = torch.cat(
        [coords, torch.ones_like(coords[:, :, [0]])], dim=-1)  

    coords_homo = torch.bmm(coords_homo, matrix.transpose(2, 1))  
    coords = (coords_homo[:, :, :2] / coords_homo[:, :, [2, 2]])  

    if warp_factor > 0:
        coords = coords / w_h * 2 - 1

        nl_part1 = coords > 1.0 - warp_factor
        nl_part2 = coords < -1.0 + warp_factor

        ret_nl_part1 = torch.tanh(
            (coords - 1.0 + warp_factor) /
            warp_factor) * warp_factor + \
            1.0 - warp_factor
        ret_nl_part2 = torch.tanh(
            (coords + 1.0 - warp_factor) /
            warp_factor) * warp_factor - \
            1.0 + warp_factor

        coords = torch.where(nl_part1, ret_nl_part1,
                             torch.where(nl_part2, ret_nl_part2, coords))

        coords = (coords + 1) / 2 * w_h

    return coords


def make_tanh_warp_grid(matrix: torch.Tensor, warp_factor: float,
                        warped_shape: Tuple[int, int],
                        orig_shape: Tuple[int, int]):
    orig_h, orig_w, *_ = orig_shape
    w_h = torch.tensor([orig_w, orig_h]).to(matrix).reshape(1, 1, 1, 2)
    return _forge_grid(
        matrix.size(0), matrix.device,
        warped_shape,
        functools.partial(inverted_tanh_warp_transform,
                          matrix=matrix,
                          warp_factor=warp_factor,
                          warped_shape=warped_shape)) / w_h*2-1


def make_inverted_tanh_warp_grid(matrix: torch.Tensor, warp_factor: float,
                                 warped_shape: Tuple[int, int],
                                 orig_shape: Tuple[int, int]):
    h, w, *_ = warped_shape
    w_h = torch.tensor([w, h]).to(matrix).reshape(1, 1, 1, 2)
    return _forge_grid(
        matrix.size(0), matrix.device,
        orig_shape,
        functools.partial(tanh_warp_transform,
                          matrix=matrix,
                          warp_factor=warp_factor,
                          warped_shape=warped_shape)) / w_h * 2-1
