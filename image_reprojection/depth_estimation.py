import cv2
import numpy as np
import torch
from transformers import pipeline
from pathlib import Path
from PIL import Image
from typing import Dict, Any

class DepthEstimator:
    def __init__(self, checkpoint: str = 'depth-anything/Depth-Anything-V2-large-hf', config: Dict[str, Any] = {}) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DepthEstimator using device: {self.device}")
        
        self.pipe = pipeline('depth-estimation', model=checkpoint, device=self.device)
        self.config = config

    def estimate_depth_from_rgb(self, img_path: Path) -> np.ndarray:
        """ Estimates relative depth map from an RGB image. """
        
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"DepthEstimator could not read {img_path}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Standard Forward Pass
        img_pil = Image.fromarray(img_rgb)
        res_std = self.pipe([img_pil]) 
        pred_std_raw = res_std[0]["predicted_depth"].squeeze().cpu().numpy()

        # Test Time Augmentation (TTA)
        if self.config.get('TTA', False):
            
            img_flipped_bgr = cv2.flip(img_bgr, 1)
            img_flipped_rgb = cv2.cvtColor(img_flipped_bgr, cv2.COLOR_BGR2RGB)
            res_flip = self.pipe(Image.fromarray(img_flipped_rgb))
            pred_flip_raw = res_flip["predicted_depth"].squeeze().cpu().numpy()
            
            # Un-flip the prediction
            pred_flip_unflipped = cv2.flip(pred_flip_raw, 1)
            
            # Average the standard and flipped passes
            pred_tensor = np.mean([pred_std_raw, pred_flip_unflipped], axis=0)
        else:
            pred_tensor = pred_std_raw

        depth_pred_linear = 1.0 / np.clip(pred_tensor, a_min=1e-6, a_max=None)

        # Smoothing
        if self.config.get('JBF', False):
            depth_src = depth_pred_linear.astype(np.float32)
            
            depth_pred_linear = cv2.ximgproc.jointBilateralFilter(
                guide=img_bgr, 
                src=depth_src, 
                d=33, 
                sigmaColor=50, 
                sigmaSpace=50
            )

        return depth_pred_linear