"""
evaluate.py - runs evaluation and computes metrics on val/test sets
"""

import os
import json
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.dataloader import get_dataloaders, get_test_loader
from src.pointnet import PointNet
from src.dgcnn import DGCNN
from src.utils import CLASS_NAMES, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    """Run model on all batches and collect predictions"""
    model.eval()
    all_preds = []
    all_labels = []

    for points, target in loader:
        points, target = points.to(device), target.to(device)
        pred, _ = model(points)
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def measure_inference_time(model, device, npoints=1024, n_runs=100):
    """Measure average single-sample inference time"""
    model.eval()
    dummy = torch.randn(1, npoints, 3).to(device)

    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    return (time.time() - t0) / n_runs


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    models = {
        'pointnet': lambda: PointNet(10),
        'dgcnn': lambda: DGCNN(10, args.k)
    }
    model = models[args.model]().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded: {args.checkpoint}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.test:
        loader = get_test_loader(args.data_root, args.npoints, args.batch_size, args.num_workers)
        eval_name = 'test'
    else:
        _, loader = get_dataloaders(args.data_root, args.npoints, args.batch_size,
                                    args.val_ratio, args.num_workers, args.seed)
        eval_name = 'val'

    preds, labels = evaluate(model, loader, device)
    # pass all label indices so classes with zero predictions still appear
    all_labels_list = list(range(len(CLASS_NAMES)))

    metrics = {
        'top1_accuracy': float(accuracy_score(labels, preds)),
        'macro_f1': float(f1_score(labels, preds, average='macro',
                                   labels=all_labels_list, zero_division=0)),
        'confusion_matrix': confusion_matrix(labels, preds, labels=all_labels_list).tolist(),
        'per_class': classification_report(labels, preds, labels=all_labels_list,
                                           target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }

    inf_time = measure_inference_time(model, device, args.npoints)
    metrics['params'] = n_params
    metrics['inference_ms'] = inf_time * 1000

    print(f"\n{'='*50}")
    print(f"  {args.model.upper()} ({eval_name.upper()}) Results")
    print(f"{'='*50}")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")

    print(f"\n{'Class':<10} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Support':<8}")
    print(f"{'-'*42}")
    for name in CLASS_NAMES:
        if name in metrics['per_class']:
            r = metrics['per_class'][name]
            print(f"{name:<10} {r['precision']:<8.3f} {r['recall']:<8.3f} "
                  f"{r['f1-score']:<8.3f} {int(r['support']):<8}")

    print(f"\nParameters:     {n_params:,}")
    print(f"Inference time: {metrics['inference_ms']:.2f} ms/sample")

    # save results
    output_dir = f'results/{args.model}'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'eval_{eval_name}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data/train_pc')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', default='pointnet', choices=['pointnet', 'dgcnn'])
    p.add_argument('--npoints', type=int, default=1024)
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test', action='store_true')
    main(p.parse_args())
