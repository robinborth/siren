from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def sample_offset(points: np.ndarray, domain: tuple[float, float] = (-0.5, 0.5)):
    _points = np.random.uniform(domain[0], domain[1], size=points.shape)
    _normals = np.ones(points.shape) * -1
    return _points, _normals


class PointCloud(Dataset):
    def __init__(self, path: str):
        super().__init__()
        pcd = o3d.io.read_point_cloud(path)
        self.coords = np.asarray(pcd.points)
        self.normals = np.asarray(pcd.normals)
        self.on_surface_points = self.coords.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        total_samples = self.coords.shape[0] * 2

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.coords.shape[0] :, :] = -1  # off-surface = -1

        off_coords, off_normals = sample_offset(self.coords)
        coords = np.concatenate((self.coords, off_coords), axis=0)
        normals = np.concatenate((self.normals, off_normals), axis=0)

        return {"coords": torch.from_numpy(coords).float()}, {
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
        }


class PointCloudCustom(Dataset):
    def __init__(self, path: str):
        super().__init__()
        pcd = o3d.io.read_point_cloud(path)
        self.coords = np.asarray(pcd.points)
        self.normals = np.asarray(pcd.normals)
        self.on_surface_points = self.coords.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        query_points, _ = sample_offset(self.coords)
        return {
            "pcd_points": torch.from_numpy(self.coords).float(),
            "pcd_normals": torch.from_numpy(self.normals).float(),
            "eikonal_points": torch.from_numpy(query_points).float(),
        }


class PointCloudItem(Dataset):
    def __init__(self, pcd_points, pcd_normals, batch_size, scan_id):
        super().__init__()

        print("Loading point cloud")
        self.coords = pcd_points.detach().cpu().numpy()
        self.normals = pcd_normals.detach().cpu().numpy()
        self.scan_id = scan_id
        print("Finished loading point cloud")

        self.on_surface_points = batch_size

    def __len__(self):
        return max(self.coords.shape[0] // self.on_surface_points, 1)

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-0.5, 0.5, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points :, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {"coords": torch.from_numpy(coords).float()}, {
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
        }


class PointCloudDataset(Dataset):
    """Simple pointcloud dataset container for multi optimization."""

    def __init__(
        self,
        data_dir: str = "/data",
        cache_id: str = "cache_id",
        n_test_size: int | None = None,
        batch_size: int = 2000,
    ):
        split = "test"
        meta_info = pd.read_csv(Path(data_dir) / "meta.csv")
        meta_info = meta_info[meta_info["split"] == split]
        meta_info = meta_info.reset_index(drop=True)
        self.split = split
        self.meta_info = meta_info
        self.cache_id = cache_id
        self.n_test_size = n_test_size
        self.batch_size = batch_size

        # load the folders of interest
        self.scan_ids = []
        self.cache_folders = []  # type ignore
        for idx, item in meta_info.iterrows():
            if (
                split == "test"
                and self.n_test_size is not None
                and len(self.cache_folders) >= self.n_test_size
            ):
                break  # skip for safe dataset settings
            scan_id = Path(item["path"]).parent.name
            path = Path(data_dir) / scan_id / self.cache_id
            self.cache_folders.append(path)
            self.scan_ids.append(scan_id)

        # preload the dataset into the cache
        self.dataset: list[dict] = []
        for idx in range(len(self.cache_folders)):
            data = self.load_eval(idx=idx)
            self.dataset.append(data)

    def load_eval(self, idx: int) -> dict:
        path = self.cache_folders[idx]
        batch = {}
        batch["pcd_points"] = torch.load(path / "eval_pcd_points.pt")
        batch["pcd_normals"] = torch.load(path / "eval_pcd_normals.pt")
        batch = {k: v.to(torch.float32) for k, v in batch.items()}
        batch["scan_id"] = path.parent.name
        return batch

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        return PointCloudItem(
            pcd_points=data["pcd_points"],
            pcd_normals=data["pcd_normals"],
            scan_id=data["scan_id"],
            batch_size=self.batch_size,
        )
