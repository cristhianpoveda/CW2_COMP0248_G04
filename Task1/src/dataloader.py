"""
dataloader.py - handles loading preprocessed point clouds with augmentation
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import set_seed


class PreprocessedPointCloud(Dataset):
    """Dataset for preprocessed .npz point clouds"""

    def __init__(self, pc_dir, augment=False, student_list=None):
        self.augment = augment
        self.data = []

        for f in sorted(os.listdir(pc_dir)):
            if not f.endswith('.npz'):
                continue
            d = np.load(os.path.join(pc_dir, f), allow_pickle=True)
            student = str(d['student'])
            if student_list is not None and student not in student_list:
                continue
            self.data.append((d['points'].astype(np.float32), int(d['label'])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        points, label = self.data[idx]
        points = points.copy()

        # unit sphere normalisation
        centroid = points.mean(axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if m > 0:
            points = points / m

        if self.augment:
            points = self._augment(points)

        return torch.from_numpy(points), torch.tensor(label, dtype=torch.long)

    def _augment(self, points):
        """Apply random augmentations to improve generalisation"""
        # rotation around Z axis
        theta = np.random.uniform(-np.pi / 7.2, np.pi / 7.2)
        c, s = np.cos(theta), np.sin(theta)
        points = points @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        angles = np.clip(0.10 * np.random.randn(3), -0.25, 0.25)
        cx, sx = np.cos(angles[0]), np.sin(angles[0])
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        cz, sz = np.cos(angles[2]), np.sin(angles[2])
        R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @
             np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
             np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])).astype(np.float32)
        points = points @ R

        # jitter to simulate sensor noise
        points += np.clip(0.01 * np.random.randn(*points.shape), -0.03, 0.03).astype(np.float32)

        points *= np.random.uniform(0.90, 1.10)
        points += np.random.uniform(-0.05, 0.05, (1, 3)).astype(np.float32)

        # randomly drop up to 15% of points to simulate partial occlusion
        dropout_ratio = np.random.random() * 0.15
        drop_idx = np.where(np.random.random(points.shape[0]) < dropout_ratio)[0]
        if len(drop_idx) > 0:
            points[drop_idx] = points[0]

        return points


def get_dataloaders(data_root, npoints=1024, batch_size=32, val_ratio=0.2,
                    num_workers=4, seed=42, **kwargs):
    """Create train/val dataloaders using student-based split"""
    set_seed(seed)

    all_students = set()
    for f in os.listdir(data_root):
        if f.endswith('.npz'):
            data = np.load(os.path.join(data_root, f), allow_pickle=True)
            all_students.add(str(data['student']))
    students = sorted(list(all_students))

    # student-based split so model never sees same person in train and val
    rng = random.Random(seed)
    rng.shuffle(students)
    n_val = max(1, int(len(students) * val_ratio))
    val_students = students[:n_val]
    train_students = students[n_val:]
    print(f"Split: {len(train_students)} train / {len(val_students)} val students")

    train_ds = PreprocessedPointCloud(data_root, augment=True, student_list=train_students)
    val_ds = PreprocessedPointCloud(data_root, augment=False, student_list=val_students)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_test_loader(data_root, npoints=1024, batch_size=32, num_workers=4, **kwargs):
    """Create test dataloader"""
    ds = PreprocessedPointCloud(data_root, augment=False)
    print(f"Test: {len(ds)} samples")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
