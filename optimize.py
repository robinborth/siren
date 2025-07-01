"""Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement."""

# Enable import from parent package
import open3d as o3d
import sys
import os
import tempfile
from siren import SirenNetwork

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sdf_meshing
from dataio import PointCloud
import training
import utils
import loss_functions

from torch.utils.data import DataLoader
from pathlib import Path

# settings
lr = 1e-4
num_epochs = 10000
resolution = 256

# folder
experiment = Path("/home/borth/siren/logs/11-49-00_pde_solver_pointcloud")
exp_folders = [p for p in experiment.iterdir() if p.is_dir()]

for exp_folder in exp_folders:

    # dataset
    point_cloud_path = str(exp_folder / "mesh/pointcloud.ply")
    dataset = PointCloud(point_cloud_path)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    # model
    model = SirenNetwork().cuda()

    # loss
    loss_fn = loss_functions.sdf
    summary_fn = utils.write_sdf_summary

    # optimize
    training.train(
        model=model,
        train_dataloader=dataloader,
        epochs=num_epochs,
        lr=lr,
        steps_til_summary=100,
        epochs_til_checkpoint=10,
        model_dir=str(exp_folder),
        loss_fn=loss_fn,
        summary_fn=summary_fn,
        double_precision=False,
        clip_grad=True,
    )

    # create the mesh
    mesh_path = exp_folder / "mesh/last.obj"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = str(Path(tmpdir) / "mesh.ply")
        sdf_meshing.create_mesh(model.net, str(tmp_path), N=resolution)
        mesh = o3d.io.read_triangle_mesh(tmp_path)
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)