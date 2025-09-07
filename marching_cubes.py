# %%
import logging
import time
from typing import Any

import numpy as np
import open3d as o3d
import plyfile
import pytorch3d
import skimage.measure
import torch
import trimesh
from pytorch3d.io import save_ply
from pytorch3d.ops.marching_cubes import marching_cubes as p3d_mc


def compute_grid(
    model,
    resolution: int = 256,
    chunk_size: int = 10_000,
    domain: tuple[float, float] = (-0.5, 0.5),
    device: str | Any = "cuda",
):
    # get the cube coordinates
    grid_vals = torch.linspace(domain[0], domain[1], resolution)
    xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
    grid = torch.stack((xs.ravel(), ys.ravel(), zs.ravel()), dim=-1).to(device)
    query = grid.reshape(resolution, resolution, resolution, 3).reshape(-1, 3)

    # evaluate the field on the grid structure
    _field = []
    for points in torch.split(query, chunk_size):
        x = model(points.to(device))
        _field.append(x.detach().cpu())
    field = torch.cat(_field).reshape(resolution, resolution, resolution, -1)

    if field.shape[-1] == 1:
        return field[..., 0]
    return field


# def marching_cubes(
#     grid: torch.Tensor,
#     isolevel: float = 0.0,
#     domain: tuple[float, float] = (-0.5, 0.5),
# ) -> pytorch3d.structures.Meshes:
#     X, Y, Z = grid.shape
#     grid = grid.permute(2, 1, 0)[None].to(torch.float32)  # (W, H, D) -> (1, D, H, W)
#     verts, faces = p3d_mc(grid, isolevel=isolevel, return_local_coords=False)
#     if not verts or not faces:
#         return pytorch3d.structures.Meshes(
#             vertices=torch.tensor([]), faces=torch.tensor([])
#         )
#     # extract the verts and the faces
#     verts = verts[0]
#     faces = faces[0]
#     # normalize verts into the domain [0, 1]
#     verts[..., 0] /= X
#     verts[..., 1] /= Y
#     verts[..., 2] /= Z
#     # normalize verts into the domain (-0.5, 0.5)
#     verts = verts * (domain[1] - domain[0]) + domain[0]

#     return pytorch3d.structures.Meshes(verts=[verts], faces=[faces])


# def marching_cubes(
#     grid: torch.Tensor,
#     isolevel: float = 0.0,
#     domain: tuple[float, float] = (-0.5, 0.5),
#     function_type: str = "sdf",  # "sdf","indicator"
# ):
#     voxel_size = 1.0 / (grid.shape[0] - 1)
#     try:
#         verts, faces, _, _ = skimage.measure.marching_cubes(
#             grid.detach().cpu().numpy(),
#             level=isolevel,
#             spacing=[voxel_size, voxel_size, voxel_size],
#             gradient_direction="descent" if function_type == "sdf" else "ascent",
#         )
#         verts = verts * (domain[1] - domain[0]) + domain[0]
#         verts = torch.tensor(verts.copy())
#         faces = torch.tensor(faces.copy())
#         return Mesh(vertices=verts, faces=faces)
#     except Exception:
#         return Mesh(vertices=torch.tensor([]), faces=torch.tensor([]))


def marching_cubes_lewiner(
    grid: torch.Tensor,
    isolevel: float = 0.0,
    domain: tuple[float, float] = (-0.5, 0.5),
    function_type: str = "sdf",  # "sdf","indicator"
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    voxel_size = 1.0 / (grid.shape[0] - 1)
    verts, faces, _, _ = skimage.measure.marching_cubes(
        grid.detach().cpu().numpy(),
        level=isolevel,
        spacing=[voxel_size, voxel_size, voxel_size],
        gradient_direction="descent" if function_type == "sdf" else "ascent",
    )
    verts = verts * (domain[1] - domain[0]) + domain[0]
    vertices = np.asarray(verts)  # Shape: (n, 3)
    faces = np.asarray(faces)  # Shape: (m, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_mesh(decoder, filename, N=256, max_batch=64**3, offset=None, scale=None):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # voxel_origin = [-1, -1, -1]
    # voxel_size = 2.0 / (N - 1)
    voxel_origin = [-0.5, -0.5, -0.5]  # domain: -0.5, 0.5
    voxel_size = 1.0 / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(sample_subset)
            .squeeze()  # .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    # convert_sdf_samples_to_ply(
    #     sdf_values.data.cpu(),
    #     voxel_origin,
    #     voxel_size,
    #     ply_filename,
    #     offset,
    #     scale,
    # )
    mesh = marching_cubes(sdf_values)
    return mesh


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = (
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros(0),
    )
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
