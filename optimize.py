"""Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import tempfile
from pathlib import Path

import open3d as o3d
from pytorch3d.io import save_obj
from torch.utils.data import DataLoader

import trainer
from dataset import PointCloud, PointCloudCustom
from marching_cubes import compute_grid, marching_cubes_lewiner
from siren import Siren, SirenNetwork

# settings
lr = 1e-4
num_epochs = 1000
resolution = 256

_model = SirenNetwork
_train_loop = trainer.train
_dataset = PointCloud

# _model = Siren
# _train_loop = trainer.train_custom
# _dataset = PointCloudCustom

# folder
# experiment = Path("/home/borth/siren/logs/17-28-32_siren_pointcloud")
# exp_folders = [p for p in experiment.iterdir() if p.is_dir()]
# exp_folders = [Path("/home/borth/siren/logs/noise=0.0,samples=300")]
path = "/home/borth/neural-poisson/logs/result/2025-09-04/08-06-10_robustness_spsr/robustness_spsr_30_scans_000/eval/bench_4273dca1b0184024b722a94c1cd50b0/pointcloud.ply"

# dataset
# point_cloud_path = str(exp_folder / "mesh/pointcloud.ply")
dataset = _dataset(path)
dataloader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=1,
    pin_memory=True,
    num_workers=0,
)

# model
model = _model().cuda()

# optimize
_train_loop(
    model=model,
    train_dataloader=dataloader,
    epochs=num_epochs,
    lr=lr,
    steps_til_summary=10,
    epochs_til_checkpoint=500,
    model_dir="/home/borth/siren/logs",
)

mesh_path = "/home/borth/siren/tmp/mesh/last.obj"
grid = compute_grid(model, resolution=resolution)
mesh = marching_cubes_lewiner(grid)
save_obj(mesh_path, mesh.verts_packed(), mesh.faces_packed())
