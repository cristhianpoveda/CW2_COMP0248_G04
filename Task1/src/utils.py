"""
utils.py - helper functions for point cloud preprocessing and dataset loading
"""

import os
import json
import random
import pickle
import numpy as np
from pathlib import Path
from functools import lru_cache
from scipy import ndimage
import torch

# class mapping
CLASS_NAMES = ['call', 'dislike', 'like', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'three']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# RealSense D455 intrinsics (640x480)
FX, FY, CX, CY = 382.444, 382.444, 319.5, 239.5


def set_seed(seed=42):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=None)
def load_metadata(meta_path):
    """Load and cache depth metadata"""
    with open(meta_path, 'r') as f:
        return json.load(f)


_detector = None

def get_detector(path='hand_detector.pkl'):
    """Load the trained RF hand detector"""
    global _detector
    if _detector is None:
        with open(path, 'rb') as f:
            _detector = pickle.load(f)
    return _detector


def extract_hand_pointcloud(depth_raw, meta_path, depth_band=0.10, detector_path='hand_detector.pkl'):
    """Extract hand point cloud from depth using RF pixel classifier"""
    meta = load_metadata(str(meta_path))
    scale = float(meta.get('depth_scale', meta.get('scale', 0.001)))
    depth_m = depth_raw.astype(np.float32) * scale
    H, W = depth_m.shape

    # bilateral filter smooths depth while preserving hand edges
    from cv2 import bilateralFilter
    depth_filtered = bilateralFilter(depth_m, d=5, sigmaColor=0.03, sigmaSpace=5)
    depth_m = np.where(depth_m > 0, depth_filtered, depth_m)

    valid = (depth_m > 0.05) & (depth_m < 3.0)
    if valid.sum() < 100:
        return np.zeros((0, 3), dtype=np.float32)

    # use RF detector to classify pixels
    rf = get_detector(detector_path)
    rows, cols = np.where(valid)

    # classify every 2nd pixel for speed then smooth the result
    step = 2
    sub_rows = rows[::step]
    sub_cols = cols[::step]

    from src.hand_detector import compute_features_batch
    feats = compute_features_batch(depth_m, sub_rows, sub_cols)
    probs = rf.predict_proba(feats)[:, 1]

    # build probability map and smooth it
    prob_map = np.zeros((H, W), dtype=np.float32)
    prob_map[sub_rows, sub_cols] = probs
    prob_map = ndimage.gaussian_filter(prob_map, sigma=3)
    hand_mask = prob_map > 0.3

    # morphological cleanup
    struct = ndimage.generate_binary_structure(2, 2)
    hand_mask = ndimage.binary_closing(hand_mask, structure=struct, iterations=2)
    hand_mask = ndimage.binary_opening(hand_mask, structure=struct, iterations=1)

    # keep only the largest connected component
    labels, n_comp = ndimage.label(hand_mask, structure=struct)
    if n_comp == 0:
        return np.zeros((0, 3), dtype=np.float32)
    comp_sizes = ndimage.sum(hand_mask, labels, range(1, n_comp + 1))
    largest = np.argmax(comp_sizes) + 1
    hand_mask = labels == largest

    final = hand_mask & valid
    rows, cols = np.where(final)
    if len(rows) < 50:
        return np.zeros((0, 3), dtype=np.float32)

    # back-project to 3D using camera intrinsics
    z = depth_m[rows, cols]
    x = (cols.astype(np.float32) - CX) * z / FX
    y = (rows.astype(np.float32) - CY) * z / FY
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def statistical_outlier_removal(points, k=20, std_ratio=2.0):
    """Remove points far from their k nearest neighbours"""
    if len(points) < k + 1:
        return points

    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    filtered = points[mean_dists < threshold]

    if len(filtered) < len(points) * 0.5:
        return points
    return filtered


def normalize_pointcloud(points):
    """Centre and scale to [-1, 1] by max extent"""
    if len(points) == 0:
        return points

    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_extent = max(
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    )
    if max_extent > 0:
        points = points / (max_extent / 2.0)
    return points


def farthest_point_sample_np(points, npoint):
    """Farthest point sampling for even spatial coverage"""
    N = points.shape[0]
    if N <= npoint:
        pad_idx = np.random.choice(N, npoint - N, replace=True)
        return np.concatenate([points, points[pad_idx]], axis=0)

    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.full(N, 1e10)
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        dist = np.sum((points - points[farthest]) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return points[centroids]


def get_dataset_samples(root_dir, student_list=None):
    """Find all depth frames in the dataset"""
    samples = []
    root = Path(root_dir)

    for student_dir in sorted(root.iterdir()):
        if not student_dir.is_dir():
            continue
        if student_list and student_dir.name not in student_list:
            continue

        subdirs = [d for d in student_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not subdirs:
            continue

        # different folder structures across students
        first = subdirs[0]
        gesture_root = student_dir if (first.name.startswith('G') and '_' in first.name) else first

        for gesture_dir in sorted(gesture_root.iterdir()):
            if not gesture_dir.is_dir():
                continue

            parts = gesture_dir.name.split('_')
            if len(parts) < 2:
                continue
            label = parts[1].lower().strip()
            if label not in CLASS_MAP:
                continue

            for clip_dir in sorted(gesture_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue

                depth_dir = clip_dir / 'depth_raw'
                meta_path = clip_dir / 'depth_metadata.json'
                if not depth_dir.exists() or not meta_path.exists():
                    continue

                for depth_file in sorted(depth_dir.glob('*.npy')):
                    samples.append({
                        'student': student_dir.name,
                        'gesture': label,
                        'depth_path': str(depth_file),
                        'meta_path': str(meta_path),
                    })
    return samples


def get_test_samples(root_dir):
    """Find all depth frames in the test set"""
    samples = []
    root = Path(root_dir)

    for gesture_dir in sorted(root.iterdir()):
        if not gesture_dir.is_dir():
            continue

        parts = gesture_dir.name.split('_')
        if len(parts) < 2:
            continue
        label = parts[1].lower().strip()
        if label not in CLASS_MAP:
            continue

        for clip_dir in sorted(gesture_dir.iterdir()):
            if not clip_dir.is_dir():
                continue

            depth_dir = clip_dir / 'depth_raw'
            meta_path = clip_dir / 'depth_metadata.json'
            if not depth_dir.exists() or not meta_path.exists():
                continue

            for depth_file in sorted(depth_dir.glob('*.npy')):
                samples.append({
                    'student': 'test',
                    'gesture': label,
                    'depth_path': str(depth_file),
                    'meta_path': str(meta_path),
                })
    return samples
