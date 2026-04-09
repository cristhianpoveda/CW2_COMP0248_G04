"""
hand_detector.py - trains a Random Forest pixel classifier to detect hands in depth images
"""

import os
import json
import argparse
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage

# RealSense D455 intrinsics
FX, FY, CX, CY = 382.444, 382.444, 319.5, 239.5


def compute_pixel_features(depth_m, row, col):
    """Compute depth-based features for a single pixel"""
    H, W = depth_m.shape
    z = depth_m[row, col]

    # local depth stats in windows of different sizes
    feats = [z]
    for win in [5, 11, 21]:
        half = win // 2
        r0, r1 = max(0, row - half), min(H, row + half + 1)
        c0, c1 = max(0, col - half), min(W, col + half + 1)
        patch = depth_m[r0:r1, c0:c1]
        valid = patch[patch > 0]
        if len(valid) > 0:
            feats.extend([valid.mean(), valid.std(), valid.min(), valid.max(), z - valid.mean()])
        else:
            feats.extend([0, 0, 0, 0, 0])

    # depth gradients
    if 0 < row < H - 1:
        dy = depth_m[row + 1, col] - depth_m[row - 1, col]
    else:
        dy = 0
    if 0 < col < W - 1:
        dx = depth_m[row, col + 1] - depth_m[row, col - 1]
    else:
        dx = 0
    feats.extend([dx, dy, np.sqrt(dx**2 + dy**2)])
    # normalised position in image
    feats.extend([row / H, col / W])

    return feats


def compute_features_batch(depth_m, rows, cols):
    """Compute depth-based features for multiple pixels using fast uniform filters"""
    H, W = depth_m.shape
    z_vals = depth_m[rows, cols]
    feats_list = [z_vals.reshape(-1, 1)]

    # local depth statistics at multiple scales
    for win in [5, 11, 21]:
        valid_mask = (depth_m > 0).astype(np.float32)
        depth_filled = np.where(depth_m > 0, depth_m, 0).astype(np.float32)

        count = ndimage.uniform_filter(valid_mask, size=win) * win * win
        count = np.maximum(count, 1)
        mean_map = ndimage.uniform_filter(depth_filled, size=win) * win * win / count
        sq_map = ndimage.uniform_filter(depth_filled**2, size=win) * win * win / count
        std_map = np.sqrt(np.maximum(sq_map - mean_map**2, 0))
        min_map = ndimage.minimum_filter(np.where(depth_m > 0, depth_m, 999), size=win)
        max_map = ndimage.maximum_filter(depth_filled, size=win)

        feats_list.append(np.column_stack([
            mean_map[rows, cols], std_map[rows, cols],
            min_map[rows, cols], max_map[rows, cols],
            z_vals - mean_map[rows, cols]
        ]))

    # larger window for broader context
    for win in [41]:
        valid_mask = (depth_m > 0).astype(np.float32)
        depth_filled = np.where(depth_m > 0, depth_m, 0).astype(np.float32)
        count = ndimage.uniform_filter(valid_mask, size=win) * win * win
        count = np.maximum(count, 1)
        mean_map = ndimage.uniform_filter(depth_filled, size=win) * win * win / count
        sq_map = ndimage.uniform_filter(depth_filled**2, size=win) * win * win / count
        std_map = np.sqrt(np.maximum(sq_map - mean_map**2, 0))
        feats_list.append(np.column_stack([
            mean_map[rows, cols], std_map[rows, cols], z_vals - mean_map[rows, cols]
        ]))

    # gradient and laplacian features
    gy = np.gradient(depth_m, axis=0)
    gx = np.gradient(depth_m, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    lap = ndimage.laplace(np.where(depth_m > 0, depth_m, 0).astype(np.float32))
    feats_list.append(np.column_stack([gx[rows, cols], gy[rows, cols], grad_mag[rows, cols], lap[rows, cols]]))

    # normalised position and distance from image centre
    cy, cx = H / 2.0, W / 2.0
    dist_centre = np.sqrt(((rows - cy) / H)**2 + ((cols - cx) / W)**2)
    feats_list.append(np.column_stack([rows / H, cols / W, dist_centre]))

    return np.hstack(feats_list).astype(np.float32)


def collect_training_data(data_root, max_samples_per_class=100000):
    """Collect hand/background pixel samples from annotated frames"""
    root = Path(data_root)
    hand_pixels = []
    bg_pixels = []

    print("Collecting training pixels from annotated frames...")
    for student_dir in sorted(root.iterdir()):
        if not student_dir.is_dir():
            continue

        subdirs = [d for d in student_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not subdirs:
            continue
        first = subdirs[0]
        gesture_root = student_dir if (first.name.startswith('G') and '_' in first.name) else first

        for gesture_dir in sorted(gesture_root.iterdir()):
            if not gesture_dir.is_dir():
                continue
            for clip_dir in sorted(gesture_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue

                ann_dir = clip_dir / 'annotation'
                depth_dir = clip_dir / 'depth_raw'
                meta_path = clip_dir / 'depth_metadata.json'
                if not ann_dir.exists() or not depth_dir.exists() or not meta_path.exists():
                    continue

                with open(meta_path) as f:
                    meta = json.load(f)
                scale = float(meta.get('depth_scale', 0.001))

                # process each annotated frame
                for ann_file in sorted(ann_dir.glob('*.png')):
                    frame_name = ann_file.stem
                    depth_file = depth_dir / f'{frame_name}.npy'
                    if not depth_file.exists():
                        continue

                    from PIL import Image
                    mask = np.array(Image.open(ann_file))
                    depth_raw = np.load(depth_file)
                    depth_m = depth_raw.astype(np.float32) * scale

                    hand_mask = mask > 127
                    valid = depth_m > 0.05

                    # hand pixels
                    hand_rows, hand_cols = np.where(hand_mask & valid)
                    if len(hand_rows) > 800:
                        idx = np.random.choice(len(hand_rows), 800, replace=False)
                        hand_rows, hand_cols = hand_rows[idx], hand_cols[idx]
                    if len(hand_rows) > 0:
                        hand_pixels.append(compute_features_batch(depth_m, hand_rows, hand_cols))

                    # subsample background more heavily since there are many more bg pixels
                    bg_rows, bg_cols = np.where(~hand_mask & valid)
                    if len(bg_rows) > 400:
                        idx = np.random.choice(len(bg_rows), 400, replace=False)
                        bg_rows, bg_cols = bg_rows[idx], bg_cols[idx]
                    if len(bg_rows) > 0:
                        bg_pixels.append(compute_features_batch(depth_m, bg_rows, bg_cols))

    hand_pixels = np.vstack(hand_pixels)
    bg_pixels = np.vstack(bg_pixels)

    # balance classes so the RF doesn't bias towards background
    n = min(len(hand_pixels), len(bg_pixels), max_samples_per_class)
    hand_idx = np.random.choice(len(hand_pixels), n, replace=False)
    bg_idx = np.random.choice(len(bg_pixels), n, replace=False)

    X = np.vstack([hand_pixels[hand_idx], bg_pixels[bg_idx]])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    # shuffle
    perm = np.random.permutation(len(X))
    print(f"Collected {n} hand + {n} background = {2*n} training pixels")
    return X[perm], y[perm]


def train_hand_detector(data_root, save_path='hand_detector.pkl'):
    """Train RF classifier then retrain on all data for the final model"""
    X, y = collect_training_data(data_root)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Validation split: {len(X_train)} train / {len(X_val)} val")

    print("Training Random Forest on 80% split...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    val_preds = rf.predict(X_val)
    print(f"\n--- Validation Results ---")
    print(f"Accuracy:  {accuracy_score(y_val, val_preds):.4f}")
    print(f"Precision: {precision_score(y_val, val_preds):.4f}")
    print(f"Recall:    {recall_score(y_val, val_preds):.4f}")
    print(f"F1:        {f1_score(y_val, val_preds):.4f}")

    hand_val = y_val == 1
    print(f"Hand pixels correct:  {(val_preds[hand_val] == 1).sum()}/{hand_val.sum()}")
    print(f"BG pixels correct:    {(val_preds[~hand_val] == 0).sum()}/{(~hand_val).sum()}")

    # retrain on all data for final model
    print("\nRetraining on 100% data for final model...")
    rf.fit(X, y)
    train_preds = rf.predict(X)
    print(f"Full training accuracy: {accuracy_score(y, train_preds):.4f}")

    with open(save_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"Saved detector to {save_path}")
    return rf


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='../rgb_depth')
    p.add_argument('--output', default='hand_detector.pkl')
    args = p.parse_args()
    train_hand_detector(args.data_root, args.output)
