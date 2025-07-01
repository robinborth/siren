#%%
from typing import Any
import torch
import open3d as o3d
from pytorch3d.ops.marching_cubes import marching_cubes as p3d_mc
import pytorch3d

device=  "cuda"
path = "/home/borth/siren/logs/experiment_1/checkpoints/model_current.pth"
model = torch.load(path, weights_only=False)

path = "/home/borth/neural-poisson/logs/ablation/2025-06-30/11-49-00_pde_solver_pointcloud/noise=0.0,samples=10000/mesh/pointcloud.ply"
pcd = o3d.io.read_point_cloud(path)


def compute_grid(self, resolution: int | None = None, chunk_size: int = 10_000,     domain: tuple[float, float] = (-0.5, 0.5),
    device: str | Any = "cpu"):
    if resolution is None:
        resolution = self.resolution

    # get the cube coordinates
    grid_vals = torch.linspace(domain[0], domain[1], resolution)
    xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
    grid = torch.stack((xs.ravel(), ys.ravel(), zs.ravel()), dim=-1).to(device)
    query = grid.reshape(resolution, resolution, resolution, 3).reshape(-1, 3)


    # evaluate the field on the grid structure
    _field = []
    for points in torch.split(query, chunk_size):
        x = model(points.to(self.device))
        _field.append(x.detach().cpu())
    field = torch.cat(_field).reshape(resolution, resolution, resolution, -1)

    if field.shape[-1] == 1:
        return field

def marching_cubes(
    grid: torch.Tensor,
    isolevel: float = 0.0,
    domain: tuple[float, float] = (-0.5, 0.5),
) ->  pytorch3d.structures.Meshes:
    X, Y, Z = grid.shape
    grid = grid.permute(2, 1, 0)[None].to(torch.float32)  # (W, H, D) -> (1, D, H, W)
    verts, faces = p3d_mc(grid, isolevel=isolevel, return_local_coords=False)
    if not verts or not faces:
        return  pytorch3d.structures.Meshes(vertices=torch.tensor([]), faces=torch.tensor([]))
    # extract the verts and the faces
    verts = verts[0]
    faces = faces[0]
    # normalize verts into the domain [0, 1]
    verts[..., 0] /= X
    verts[..., 1] /= Y
    verts[..., 2] /= Z
    # normalize verts into the domain (-0.5, 0.5)
    verts = verts * (domain[1] - domain[0]) + domain[0]
    return  pytorch3d.structures.Meshes(vertices=verts, faces=faces)
