import numpy as np
import torch
import torchvision
from einops import rearrange, repeat


def get_position_map_from_depth(depth, mask, intrinsics, extrinsics, image_wh=None):
    """Compute the position map from the depth map and the camera parameters for a batch of views.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        intrinsics (torch.Tensor): The camera intrinsics matrices with the shape (B, 3, 3).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        image_wh (Tuple[int, int]): The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    #print(f"\n[DEBUG] get_position_map_from_depth inputs:")
    #print(f"  depth: shape={depth.shape}, min={depth.min()}, max={depth.max()}, has_nan={torch.isnan(depth).any()}")
    #print(f"  mask: shape={mask.shape}, sum={mask.sum()}, has_nan={torch.isnan(mask).any()}")
    #print(f"  intrinsics: shape={intrinsics.shape}, has_nan={torch.isnan(intrinsics).any()}")
    #print(f"  extrinsics: shape={extrinsics.shape}, has_nan={torch.isnan(extrinsics).any()}")
    
    # Print sample intrinsics to check for zeros in focal length
    """if intrinsics.shape[0] > 0:
        print(f"  Sample intrinsics[0]:\n{intrinsics[0]}")
        print(f"  Focal lengths (fx, fy): ({intrinsics[0, 0, 0]}, {intrinsics[0, 1, 1]})")
        if intrinsics[0, 0, 0] == 0 or intrinsics[0, 1, 1] == 0:
            print("[ERROR] Zero focal length detected!")"""
    
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)
    
    #print(f"  Squeezed depth: shape={depth.shape}, min={depth.min()}, max={depth.max()}, has_nan={torch.isnan(depth).any()}")

    u_coord, v_coord = torch.meshgrid(
        torch.arange(image_wh[0]), torch.arange(image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Check focal lengths before division
    fx = intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1)
    
    #print(f"  fx: min={fx.min()}, max={fx.max()}, has_zero={(fx == 0).any()}")
    #print(f"  fy: min={fy.min()}, max={fy.max()}, has_zero={(fy == 0).any()}")
    
    # Compute the position map by back-projecting depth pixels to 3D space
    x = (u_coord - cx) * depth / fx
    y = (v_coord - cy) * depth / fy
    z = depth
    
    #print(f"  After unprojection:")
    #print(f"    x: min={x.min()}, max={x.max()}, has_nan={torch.isnan(x).any()}")
    #print(f"    y: min={y.min()}, max={y.max()}, has_nan={torch.isnan(y).any()}")
    #print(f"    z: min={z.min()}, max={z.max()}, has_nan={torch.isnan(z).any()}")

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)
    #print(f"  camera_coords: shape={camera_coords.shape}, has_nan={torch.isnan(camera_coords).any()}")

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    #print(f"  coords_homogeneous: shape={coords_homogeneous.shape}, has_nan={torch.isnan(coords_homogeneous).any()}")
    

    #here it stops) 
    # Check extrinsics for invertibility
    """if extrinsics.shape[0] > 0:
        det = torch.det(extrinsics[0, :3, :3])
        print(f"  Extrinsics[0] determinant: {det}")
        if abs(det) < 1e-6:
            print("[ERROR] Near-singular extrinsics matrix!")"""
    
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)
    
    #print(f"  world_coords: shape={world_coords.shape}, has_nan={torch.isnan(world_coords).any()}")
    #print(f"  world_coords: min={world_coords.min()}, max={world_coords.max()}")

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask
    
    #print(f"  Final position_map: shape={position_map.shape}, has_nan={torch.isnan(position_map).any()}")
    #print(f"  Final position_map: min={position_map.min()}, max={position_map.max()}")
    
    # If NaN detected, show where
    """if torch.isnan(position_map).any():
        nan_locations = torch.where(torch.isnan(position_map))
        print(f"  NaN locations (first 10): {[loc[:10].tolist() for loc in nan_locations]}")"""

    return position_map


def get_position_map(
    depth,
    cam2world_matrix,
    intrinsics,
    resolution,
    scale=0.001,
    offset=0.5,
):
    """
    Compute the position map from the depth map and the camera parameters for a batch of views.

    depth: (B, F, 1, H, W)
    cam2world_matrix: (B, F, 4, 4)
    intrinsics: (B, F, 3, 3)
    resolution: int
    """
    #print(f"\n[DEBUG] get_position_map inputs:")
    #print(f"  depth: shape={depth.shape}, min={depth.min()}, max={depth.max()}, has_nan={torch.isnan(depth).any()}")
    #print(f"  cam2world_matrix: shape={cam2world_matrix.shape}, has_nan={torch.isnan(cam2world_matrix).any()}")
    #print(f"  intrinsics: shape={intrinsics.shape}, has_nan={torch.isnan(intrinsics).any()}")
    #print(f"  scale={scale}, offset={offset}")
    
    bsz = depth.shape[0]
    depths = rearrange(depth, "b f c h w -> (b f) h w c").to(
        dtype=cam2world_matrix.dtype
    )
    masks = depths > 0
    
    #print(f"  Rearranged depths: shape={depths.shape}, has_nan={torch.isnan(depths).any()}")
    #print(f"  Mask coverage: {masks.float().mean().item():.4f} (fraction of positive depths)")
    
    cam2world_matrices = rearrange(cam2world_matrix, "b f c1 c2 -> (b f) c1 c2")
    intrinsics = rearrange(intrinsics, "b f c1 c2 -> (b f) c1 c2")
    
    #print(f"  Rearranged cam2world_matrices: shape={cam2world_matrices.shape}")
    #print(f"  Rearranged intrinsics: shape={intrinsics.shape}")

    position_maps = get_position_map_from_depth(
        depths, masks, intrinsics, cam2world_matrices
    )
    
    #print(f"\n[DEBUG] After get_position_map_from_depth:")
    #print(f"  position_maps: shape={position_maps.shape}, has_nan={torch.isnan(position_maps).any()}")
    #print(f"  position_maps: min={position_maps.min()}, max={position_maps.max()}")

    # Convert to meters and clamp values
    position_maps_scaled = position_maps * scale + offset
    #print(f"  After scaling (before clamp): min={position_maps_scaled.min()}, max={position_maps_scaled.max()}")
    
    position_maps = position_maps_scaled.clamp(0.0, 1.0)
    #print(f"  After clamping: min={position_maps.min()}, max={position_maps.max()}, has_nan={torch.isnan(position_maps).any()}")

    position_maps = rearrange(position_maps, "(b f) h w c -> b f c h w", b=bsz)
    #print(f"  Final rearranged: shape={position_maps.shape}, has_nan={torch.isnan(position_maps).any()}")

    return position_maps