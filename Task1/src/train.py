"""
train.py - training script for point cloud gesture classification
"""

import os
import time
import json
import argparse
import torch
import torch.optim as optim
from src.dataloader import get_dataloaders
from src.pointnet import PointNet
from src.dgcnn import DGCNN
from src.utils import set_seed


def get_model(name, num_classes=10, k=20):
    """Create model by name"""
    if name == 'pointnet':
        return PointNet(num_classes=num_classes)
    elif name == 'dgcnn':
        return DGCNN(num_classes=num_classes, k=k)
    raise ValueError(f"Unknown model: {name}")


def train_epoch(model, loader, optimizer, device):
    """Run one training epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for points, target in loader:
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        pred, tf = model(points)
        loss = model.get_loss(pred, target, tf)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * target.size(0)
        correct += (pred.argmax(1) == target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    """Run validation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for points, target in loader:
        points, target = points.to(device), target.to(device)
        pred, tf = model(points)
        loss = model.get_loss(pred, target, tf)
        total_loss += loss.item() * target.size(0)
        correct += (pred.argmax(1) == target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


def main(args):
    # DGCNN works better with higher lr + SGD, PointNet with lower lr + Adam
    if args.lr is None:
        args.lr = 0.05 if args.model == 'dgcnn' else 0.001

    print(f"Training {args.model} | lr={args.lr}")
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(
        args.data_root, args.npoints, args.batch_size,
        args.val_ratio, args.num_workers, args.seed
    )

    model = get_model(args.model, 10, args.k).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.model == 'dgcnn':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs('weights', exist_ok=True)
    os.makedirs(f'results/{args.model}', exist_ok=True)

    best_val_acc = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer, device)
        vl, va = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {tl:.4f} / {ta:.4f} | "
              f"Val: {vl:.4f} / {va:.4f} | "
              f"LR: {lr:.6f} | {time.time()-t0:.0f}s")

        history.append({
            'epoch': epoch,
            'train_loss': tl, 'train_acc': ta,
            'val_loss': vl, 'val_acc': va,
            'lr': lr
        })

        # save checkpoint when validation accuracy improves
        if va > best_val_acc:
            best_val_acc = va
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': va,
                'args': vars(args)
            }, f'weights/{args.model}.pth')
            print(f"  -> saved (val_acc: {va:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(f'results/{args.model}/history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    print(f"Weights: weights/{args.model}.pth")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data/train_pc')
    p.add_argument('--model', default='pointnet', choices=['pointnet', 'dgcnn'])
    p.add_argument('--npoints', type=int, default=1024)
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=20)
    main(p.parse_args())
