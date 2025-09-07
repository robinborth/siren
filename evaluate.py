"""Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement."""

# Enable import from parent package
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pathlib import Path

import configargparse
import torch
import yaml
from torch.utils.data import DataLoader

import loss_functions
import modules
import training
import utils
from dataset import PointCloudDataset
from marching_cubes import compute_grid, marching_cubes_lewiner


class SDFDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, coords):
        model_in = {"coords": coords}
        return self.model(model_in)["model_out"]


ROOT = Path("/home/borth/siren/")

# load the config
p = configargparse.ArgumentParser()
p.add(
    "-c",
    "--config",
    required=True,
    help="Path to config file.",
)
opt = p.parse_args()
c_name = opt.config
path = ROOT / "config" / c_name
with open(path, "r") as f:
    config = yaml.safe_load(f)

datamodule = PointCloudDataset(
    data_dir=config["data_dir"],
    cache_id=config["cache_id"],
    n_test_size=config["n_test_size"],
    batch_size=config["batch_size"],
)
for idx in range(len(datamodule)):
    dataset = datamodule[idx]
    scan_id = dataset.scan_id
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        pin_memory=True,
        num_workers=0,
    )

    # Define the model.
    model = modules.SingleBVPNet(type="sine", in_features=3)
    model.cuda()

    # Define the loss
    loss_fn = loss_functions.sdf
    summary_fn = utils.write_sdf_summary

    train_dir = ROOT / "train" / config["experiment_name"] / scan_id
    train_dir.parent.mkdir(parents=True, exist_ok=True)
    training.train(
        model=model,
        train_dataloader=dataloader,
        epochs=config["num_epochs"],
        lr=float(config["lr"]),
        steps_til_summary=config["steps_til_summary"],
        epochs_til_checkpoint=config["epochs_til_ckpt"],
        model_dir=train_dir,
        loss_fn=loss_fn,
        summary_fn=summary_fn,
        double_precision=False,
        clip_grad=True,
    )

    import_dir = Path("/home/borth/neural-poisson/import/siren")
    mesh_path = import_dir / config["experiment_name"] / f"{scan_id}.ply"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    grid = compute_grid(SDFDecoder(model=model), resolution=config["resolution"])
    mesh = marching_cubes_lewiner(grid)
    mesh.export(mesh_path)
