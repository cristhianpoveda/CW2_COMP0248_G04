import torch
import cv2
import numpy as np
import kornia
from pathlib import Path
from typing import Tuple, Optional

class TensorLoader:
    def __init__(self, K: np.ndarray, D: np.ndarray, target_res: Tuple[int, int] = (640, 480), device: torch.device = torch.device('cpu')) -> None:
        self.K_orig: np.ndarray = np.array(K, dtype=np.float32)
        self.D: np.ndarray = np.array(D, dtype=np.float32)
        self.target_res: Tuple[int, int] = target_res
        self.device: torch.device = device
        
        self.K_rescaled: Optional[np.ndarray] = None

    def _undistort_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self.K_orig, self.D)

    def _rescale_calibration_matrix(self, orig_h: int, orig_w: int) -> np.ndarray:
        """Scales the intrinsic matrix according to the new resolution."""
        target_w, target_h = self.target_res
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        K_new = self.K_orig.copy()
        K_new[0, 0] *= scale_x  # fx
        K_new[1, 1] *= scale_y  # fy
        K_new[0, 2] *= scale_x  # cx
        K_new[1, 2] *= scale_y  # cy
        
        self.K_rescaled = K_new
        return K_new

    def load_img_to_device_as_tensor(self, img_path: Path) -> torch.Tensor:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        orig_h, orig_w = img.shape
        
        img_undistorted = self._undistort_image(img)
        
        img_resized = cv2.resize(img_undistorted, self.target_res, interpolation=cv2.INTER_AREA)
        if self.K_rescaled is None:
            self._rescale_calibration_matrix(orig_h, orig_w)
            
        # Convert to PyTorch Tensor [B, C, H, W] normalized between [0, 1]
        img_tensor = kornia.utils.image_to_tensor(img_resized, keepdim=False).float() / 255.0
        
        return img_tensor.to(self.device)
