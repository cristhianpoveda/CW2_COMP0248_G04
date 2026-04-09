import os
import cv2
import numpy as np
import torch
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from diffusers import MarigoldDepthPipeline
from tqdm import tqdm

# basic setup
device = "cuda:3" 
base_dirs = [
    '/cs/student/project_msc/2025/rai/mdecastr/Object Detection/coursework2/Group04_v2/Sequence_A',
    '/cs/student/project_msc/2025/rai/mdecastr/Object Detection/coursework2/Group04_v2/Sequence_B'
]
output_dir = 'output_results'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'depth_maps'), exist_ok=True)

# gather valid image pairs
image_pairs = {}
for b_dir in base_dirs:
    seq_name = os.path.basename(b_dir)
    rgb_dir = os.path.join(b_dir, 'camera_color_image_raw')
    depth_dir = os.path.join(b_dir, 'camera_aligned_depth_to_color_image_raw')
    
    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
        print(f"Skipping {seq_name}: missing directories.")
        continue
        
    for f in os.listdir(rgb_dir):
        if f.endswith('.png'):
            d_path = os.path.join(depth_dir, f)
            if os.path.exists(d_path):
                img_key = f"{seq_name}/{f}"
                image_pairs[img_key] = (os.path.join(rgb_dir, f), d_path)

print(f"Found {len(image_pairs)} valid image pairs.")

models_list = ['Depth Anything V2', 'ZoeDepth', 'Marigold LCM']
metrics = {m: {} for m in models_list}
predictions = {m: {} for m in models_list}

# helpers
def load_gt_and_mask(depth_gt_path):
    # read real sense depth, convert to meters, and filter invalid ranges
    depth_gt = cv2.imread(depth_gt_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
    mask = (depth_gt > 0.01) & (depth_gt < 15.0)
    valid_gt = depth_gt[mask]
    valid_gt_safe = np.maximum(valid_gt, 1e-5)
    return depth_gt, mask, valid_gt, valid_gt_safe

def get_metrics(pred, mask, valid_gt, valid_gt_safe):
    valid_pred = pred[mask]
    abs_rel = float(np.mean(np.abs(valid_gt_safe - valid_pred) / valid_gt_safe))
    rmse = float(np.sqrt(np.mean((valid_gt - valid_pred) ** 2)))
    ratio = np.maximum(valid_gt_safe / valid_pred, valid_pred / valid_gt_safe)
    d1 = float(np.mean(ratio < 1.25))
    return abs_rel, rmse, d1

def save_depth_viz(pred_map, model_name, img_key):
    # save a quick visual representation of the predicted depth map
    seq, fname = img_key.split('/')
    save_path = os.path.join(output_dir, 'depth_maps', f"{seq}_{model_name}_{fname}")
    plt.imsave(save_path, pred_map, cmap='plasma', vmin=0, vmax=5.0)

# run model inference sequentially to manage vram
# 1. Depth Anything V2
print("\nRunning Depth Anything V2...")
pipe_da = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-large-hf", device=device)
for img_key, (rgb_path, depth_path) in tqdm(image_pairs.items()):
    img_pil = Image.open(rgb_path).convert("RGB")
    depth_gt, mask, valid_gt, valid_gt_safe = load_gt_and_mask(depth_path)
    gt_h, gt_w = depth_gt.shape
    
    pred_da = pipe_da(img_pil)["predicted_depth"].squeeze().cpu().numpy()
    pred_da = cv2.resize(pred_da, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
    
    # inverse depth scaling
    pred_da = np.maximum(pred_da, 1e-5)
    pred_da_linear = 1.0 / pred_da
    scale_da = np.median(valid_gt) / np.median(pred_da_linear[mask])
    aligned = np.clip(pred_da_linear * scale_da, 0.01, 15.0)

    predictions['Depth Anything V2'][img_key] = aligned
    abs_rel, rmse, d1 = get_metrics(aligned, mask, valid_gt, valid_gt_safe)
    metrics['Depth Anything V2'][img_key] = {'abs_rel': abs_rel, 'rmse': rmse, 'd1': d1}
    save_depth_viz(aligned, 'DAv2', img_key)

del pipe_da
torch.cuda.empty_cache()

# 2. ZoeDepth
print("\nRunning ZoeDepth...")
zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True).to(device)
zoe.eval()
for img_key, (rgb_path, depth_path) in tqdm(image_pairs.items()):
    img_pil = Image.open(rgb_path).convert("RGB")
    depth_gt, mask, valid_gt, valid_gt_safe = load_gt_and_mask(depth_path)
    gt_h, gt_w = depth_gt.shape
    
    with torch.no_grad():
        pred_zoe = zoe.infer_pil(img_pil)

    aligned = cv2.resize(pred_zoe, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
    aligned = np.clip(aligned, 0.01, 15.0)

    predictions['ZoeDepth'][img_key] = aligned
    abs_rel, rmse, d1 = get_metrics(aligned, mask, valid_gt, valid_gt_safe)
    metrics['ZoeDepth'][img_key] = {'abs_rel': abs_rel, 'rmse': rmse, 'd1': d1}
    save_depth_viz(aligned, 'Zoe', img_key)

del zoe
torch.cuda.empty_cache()

# 3. Marigold LCM
print("\nRunning Marigold LCM...")
pipe_mg = MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to(device)

for img_key, (rgb_path, depth_path) in tqdm(image_pairs.items()):
    img_pil = Image.open(rgb_path).convert("RGB")
    depth_gt, mask, valid_gt, valid_gt_safe = load_gt_and_mask(depth_path)
    gt_h, gt_w = depth_gt.shape
    
    pred_mg = pipe_mg(img_pil).prediction
    pred_mg = pred_mg.squeeze().cpu().numpy() if torch.is_tensor(pred_mg) else pred_mg.squeeze()
        
    pred_mg = cv2.resize(pred_mg, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
    scale_mg = np.median(valid_gt) / np.median(pred_mg[mask])
    aligned = np.clip(pred_mg * scale_mg, 0.01, 15.0)

    predictions['Marigold LCM'][img_key] = aligned
    abs_rel, rmse, d1 = get_metrics(aligned, mask, valid_gt, valid_gt_safe)
    metrics['Marigold LCM'][img_key] = {'abs_rel': abs_rel, 'rmse': rmse, 'd1': d1}
    save_depth_viz(aligned, 'Marigold', img_key)

del pipe_mg
torch.cuda.empty_cache()

# dump metrics
metrics_path = os.path.join(output_dir, 'depth_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"\nSaved metrics to {metrics_path}")

# format data for pandas/seaborn plotting
records = []
for model, imgs in metrics.items():
    for img_name, mets in imgs.items():
        records.append({
            'Model': model,
            'Image': img_name,
            'AbsRel': mets['abs_rel'],
            'RMSE': mets['rmse'],
            'd1': mets['d1'] * 100 
        })
df = pd.DataFrame(records)

# plotting styles
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = ["#3498db", "#e67e22", "#2ecc71"]

# 1. mean metric bar charts
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
sns.barplot(data=df, x='Model', y='AbsRel', ax=axs[0], palette=palette, edgecolor='black')
axs[0].set_title('Mean AbsRel (Lower is Better)')

sns.barplot(data=df, x='Model', y='RMSE', ax=axs[1], palette=palette, edgecolor='black')
axs[1].set_title('Mean RMSE (Lower is Better)')

sns.barplot(data=df, x='Model', y='d1', ax=axs[2], palette=palette, edgecolor='black')
axs[2].set_title('Mean d1 Accuracy (Higher is Better)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_metrics_bars.png'), dpi=300)
plt.close()

# 2. error distribution (boxplot)
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Model', y='AbsRel', ax=ax, palette=palette, linewidth=2)
ax.set_title('AbsRel Error Distribution')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'error_distribution_boxplot.png'), dpi=300)
plt.close()

# 3. generate error maps for the WORST zoe prediction 
worst_zoe_img = max(metrics['ZoeDepth'], key=lambda k: metrics['ZoeDepth'][k]['abs_rel'])
rgb_path, depth_path = image_pairs[worst_zoe_img]
depth_gt, mask, _, _ = load_gt_and_mask(depth_path)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
img_pil = Image.open(rgb_path).convert("RGB")

axs[0, 0].imshow(np.array(img_pil))
axs[0, 0].set_title("Original RGB")
axs[0, 0].axis('off')

# plot error maps for the other quadrants
plot_coords = [(0, 1), (1, 0), (1, 1)]
for model, coord in zip(models_list, plot_coords):
    err = np.abs(predictions[model][worst_zoe_img] - depth_gt)
    err[~mask] = np.nan # hide invalid pixels
    
    im = axs[coord].imshow(err, cmap='magma', vmin=0, vmax=2.0)
    axs[coord].set_title(f"{model} Error")
    axs[coord].axis('off')
    fig.colorbar(im, ax=axs[coord], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'worst_zoe_error_maps.png'), dpi=300)
plt.close()

# 4. generate depth maps for the BEST zoe prediction
best_zoe_img = min(metrics['ZoeDepth'], key=lambda k: metrics['ZoeDepth'][k]['abs_rel'])
_, best_depth_path = image_pairs[best_zoe_img]
depth_gt, mask, _, _ = load_gt_and_mask(best_depth_path)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# prep ground truth viz (masking out the invalid spots so they appear blank/white)
gt_viz = np.copy(depth_gt)
gt_viz[~mask] = np.nan

im_gt = axs[0, 0].imshow(gt_viz, cmap='plasma', vmin=0, vmax=5.0)
axs[0, 0].set_title("Ground Truth")
axs[0, 0].axis('off')
fig.colorbar(im_gt, ax=axs[0, 0], fraction=0.046, pad=0.04)

for model, coord in zip(models_list, plot_coords):
    pred_viz = np.copy(predictions[model][best_zoe_img])
    pred_viz[~mask] = np.nan 
    
    im = axs[coord].imshow(pred_viz, cmap='plasma', vmin=0, vmax=5.0)
    axs[coord].set_title(f"{model} Prediction")
    axs[coord].axis('off')
    fig.colorbar(im, ax=axs[coord], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'best_zoe_depth_maps.png'), dpi=300)
plt.close()

print("\nProcessing complete. All charts and depth maps saved to the output directory.")