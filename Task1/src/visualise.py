"""
visualise.py - generates plots and figures for the report
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils import CLASS_NAMES


def plot_confusion_matrix(eval_path, output_dir, model_name, suffix):
    """Plot confusion matrix with count annotations"""
    with open(eval_path) as f:
        data = json.load(f)

    cm = np.array(data['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name.upper()} Confusion Matrix')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cm_{model_name}{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history_path, output_dir, model_name):
    """Plot loss and accuracy curves over training"""
    with open(history_path) as f:
        h = json.load(f)

    epochs = [x['epoch'] for x in h]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))

    a1.plot(epochs, [x['train_loss'] for x in h], 'b-', label='Train', lw=1.5)
    a1.plot(epochs, [x['val_loss'] for x in h], 'r-', label='Val', lw=1.5)
    a1.set_xlabel('Epoch')
    a1.set_ylabel('Loss')
    a1.set_title(f'{model_name.upper()} - Loss')
    a1.legend()
    a1.grid(True, alpha=0.3)

    a2.plot(epochs, [x['train_acc'] for x in h], 'b-', label='Train', lw=1.5)
    a2.plot(epochs, [x['val_acc'] for x in h], 'r-', label='Val', lw=1.5)
    a2.set_xlabel('Epoch')
    a2.set_ylabel('Accuracy')
    a2.set_title(f'{model_name.upper()} - Accuracy')
    a2.legend()
    a2.grid(True, alpha=0.3)
    a2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'curves_{model_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(eval_paths, output_dir, suffix):
    """Plot per-class F1 comparison between models"""
    data = {}
    for name, path in eval_paths.items():
        with open(path) as f:
            data[name] = json.load(f)

    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(data)
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#9B59B6']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, d) in enumerate(data.items()):
        f1s = [d['per_class'][c]['f1-score'] for c in CLASS_NAMES]
        ax.bar(x + i * w - 0.4 + w / 2, f1s, w, label=name.upper(),
               color=colors[i % len(colors)], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Per-Class F1 Score Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'f1_comparison{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _select_best_samples(pc_dir):
    """Pick best-looking point cloud per gesture for the figure"""
    all_gestures = {}
    for f in sorted(os.listdir(pc_dir)):
        if not f.endswith('.npz'):
            continue
        data = np.load(os.path.join(pc_dir, f), allow_pickle=True)
        g = str(data['gesture'])
        if g not in all_gestures:
            all_gestures[g] = []
        if len(all_gestures[g]) < 2000:
            all_gestures[g].append(data['points'])

    # these gestures don't look good with auto scoring
    manual = {'like': 17, 'ok': 18}

    best = {}
    for g, pts_list in all_gestures.items():
        if g in manual:
            step = max(1, len(pts_list) // 20)
            idx = manual[g] * step
            if idx < len(pts_list):
                best[g] = pts_list[idx]
                continue

        best_score = -1
        best_pts = pts_list[0]
        for pts in pts_list:
            ext = pts.max(axis=0) - pts.min(axis=0)
            if ext.min() < 0.15:
                continue
            height_width = ext[1] / max(ext[0], 0.01)
            if height_width > 3.0:
                continue
            score = ext[0] * ext[1] * ext[2] * min(height_width, 2.0)
            if score > best_score:
                best_score = score
                best_pts = pts
        best[g] = best_pts

    return best


def plot_point_cloud_samples(pc_dir, output_dir):
    """Plot all 10 gestures as 3D point clouds"""
    best = _select_best_samples(pc_dir)
    if not best:
        return

    fig = plt.figure(figsize=(24, 10))
    fig.patch.set_facecolor('white')

    for i, name in enumerate(CLASS_NAMES):
        if name not in best:
            continue
        pts = best[name]
        ax = fig.add_subplot(2, 5, i + 1, projection='3d')
        # swap Y and Z axes so the hand faces the viewer naturally
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
                   c=pts[:, 1], cmap='plasma', s=4, alpha=0.85, edgecolors='none')
        ax.set_title(name.upper(), fontsize=14, fontweight='bold', pad=12)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#dddddd')
        ax.yaxis.pane.set_edgecolor('#dddddd')
        ax.zaxis.pane.set_edgecolor('#dddddd')
        ax.grid(False)
        ax.view_init(elev=20, azim=135)

    plt.suptitle('Hand Gesture Point Clouds (1024 points)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'point_cloud_samples.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Point cloud samples saved")


def main(args):
    fig_dir = os.path.join('results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print("Generating figures...")

    suffix = '_test' if args.test else '_val'
    eval_file = f'eval_{suffix.lstrip("_")}.json'

    for m in ['pointnet', 'dgcnn']:
        ep = os.path.join('results', m, eval_file)
        hp = os.path.join('results', m, 'history.json')
        if os.path.exists(ep):
            print(f"  Confusion matrix: {m}")
            plot_confusion_matrix(ep, fig_dir, m, suffix)
        if os.path.exists(hp):
            print(f"  Training curves: {m}")
            plot_training_curves(hp, fig_dir, m)

    eval_paths = {}
    for m in ['pointnet', 'dgcnn']:
        ep = os.path.join('results', m, eval_file)
        if os.path.exists(ep):
            eval_paths[m] = ep
    if len(eval_paths) >= 2:
        print("  Comparison chart")
        plot_comparison_bar(eval_paths, fig_dir, suffix)

    if os.path.exists('data/train_pc'):
        plot_point_cloud_samples('data/train_pc', fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true')
    main(p.parse_args())
