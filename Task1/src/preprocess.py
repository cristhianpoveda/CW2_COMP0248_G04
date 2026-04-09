"""
preprocess.py - converts raw depth maps to point clouds using RF hand detector
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from src.utils import (get_dataset_samples, get_test_samples,
                       extract_hand_pointcloud, statistical_outlier_removal,
                       normalize_pointcloud, farthest_point_sample_np, CLASS_MAP)


def preprocess_samples(samples, output_dir, npoints=1024, detector_path='hand_detector.pkl'):
    """Convert depth frames to point clouds"""
    os.makedirs(output_dir, exist_ok=True)
    skipped = 0

    for i, s in enumerate(tqdm(samples, desc="Converting depth -> point cloud")):
        depth = np.load(s['depth_path'], allow_pickle=True)
        hand = extract_hand_pointcloud(depth, s['meta_path'], detector_path=detector_path)

        # replace bad frames with noise so we keep the label in the dataset
        if len(hand) < 10:
            hand = np.random.randn(npoints, 3).astype(np.float32) * 0.01
            skipped += 1
        else:
            hand = statistical_outlier_removal(hand, k=20, std_ratio=2.0)
            hand = normalize_pointcloud(hand)
            hand = farthest_point_sample_np(hand, npoints)

        np.savez_compressed(
            os.path.join(output_dir, f'{i:05d}.npz'),
            points=hand.astype(np.float32),
            label=np.int64(CLASS_MAP[s['gesture']]),
            gesture=s['gesture'],
            student=s.get('student', 'test'),
        )

    print(f"Saved {len(samples)} samples to {output_dir} "
          f"({skipped} bad frames replaced with noise)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='../rgb_depth')
    p.add_argument('--test_root', default='../test_data')
    p.add_argument('--output_train', default='data/train_pc')
    p.add_argument('--output_test', default='data/test_pc')
    p.add_argument('--npoints', type=int, default=1024)
    p.add_argument('--detector', default='hand_detector.pkl')
    args = p.parse_args()

    print(f"Using detector: {args.detector}")

    if os.path.exists(args.data_root):
        samples = get_dataset_samples(args.data_root)
        print(f"Found {len(samples)} training samples")
        preprocess_samples(samples, args.output_train, args.npoints, args.detector)

    if args.test_root and os.path.exists(args.test_root):
        test_samples = get_test_samples(args.test_root)
        print(f"Found {len(test_samples)} test samples")
        preprocess_samples(test_samples, args.output_test, args.npoints, args.detector)

    print("Done.")
