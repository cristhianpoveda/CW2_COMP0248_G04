import torch
import cv2
import numpy as np
from kornia.feature import LoFTR
from pathlib import Path
from scipy.spatial import KDTree
from typing import Tuple, Dict, Any, List, Optional
from tensor_loader import TensorLoader
from depth_estimation import DepthEstimator

class PoseEstimation:
    def __init__(self, K: np.ndarray, D: np.ndarray, method: str = 'prosac', config: Dict[str, Any] = {}) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.config: Dict[str, Any] = config
        
        self.matcher = LoFTR(pretrained=self.config['model']['weights']).to(self.device).eval()
        self.loader = TensorLoader(K, D, tuple(self.config['model']['target_res']), self.device)
        self.depth_estimator = DepthEstimator(config=self.config['depth_estimation'])
        self.method: str = method
        self._tensor_cache: Dict[Path, torch.Tensor] = {}

        self.prev_3d_points: Optional[np.ndarray] = None
        self.prev_2d_points_f2: Optional[np.ndarray] = None
        self.prev_pose_R: Optional[np.ndarray] = None
        self.prev_pose_t: Optional[np.ndarray] = None

    def _get_frame_data(self, img_path: Path, require_depth: bool = False) -> dict:
        """ Cache manager for LoFTR tensors and Depth maps """
        if img_path not in self._tensor_cache:
            
            if len(self._tensor_cache) >= 3:
                oldest_key = list(self._tensor_cache.keys())[0]
                del self._tensor_cache[oldest_key]
            
            frame_data = {
                'tensor': self.loader.load_img_to_device_as_tensor(img_path),
                'depth_map': None
            }
            self._tensor_cache[img_path] = frame_data
            
        if require_depth and self._tensor_cache[img_path]['depth_map'] is None:
            self._tensor_cache[img_path]['depth_map'] = self.depth_estimator.estimate_depth_from_rgb(img_path)
            
        return self._tensor_cache[img_path]
    
    def _get_correspondences_img_pair(self, img_path_1: Path, img_path_2: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img0_tensor = self._get_frame_data(img_path_1, require_depth=False)['tensor']
        img1_tensor = self._get_frame_data(img_path_2, require_depth=False)['tensor']

        input_dict = {"image0": img0_tensor, "image1": img1_tensor}
        
        with torch.no_grad():
            correspondences = self.matcher(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()

        return mkpts0, mkpts1, confidence
    
    def _get_filtered_correspondences_3_frames(self, img_path_1: Path, img_path_2: Path, img_path_3: Path) -> Tuple[np.ndarray, np.ndarray]:
        mkpts1_12, mkpts2_12, _ = self._get_correspondences_img_pair(img_path_1, img_path_2)
        mkpts1_13, mkpts3_13, _ = self._get_correspondences_img_pair(img_path_1, img_path_3)
        mkpts2_23, mkpts3_23, _ = self._get_correspondences_img_pair(img_path_2, img_path_3)

        valid_indices_12: List[int] = []

        tree_23 = KDTree(mkpts2_23)
        tree_13 = KDTree(mkpts3_13)

        distances_2, indices_23 = tree_23.query(mkpts2_12, distance_upper_bound=self.config['feature_matching']['kdtree_threshold'])

        for i, (dist2, j) in enumerate(zip(distances_2, indices_23)):
            if dist2 == float('inf'):
                continue

            candidate_p3 = mkpts3_23[j]
            dist3, k = tree_13.query(candidate_p3, distance_upper_bound=self.config['feature_matching']['kdtree_threshold'])

            if dist3 == float('inf'):
                continue

            original_p1 = mkpts1_12[i]
            closed_p1 = mkpts1_13[k]

            dist1 = np.linalg.norm(original_p1 - closed_p1)

            if dist1 <= self.config['feature_matching']['kdtree_threshold']:
                valid_indices_12.append(i)

        filtered_mkpts1 = mkpts1_12[valid_indices_12]
        filtered_mkpts2 = mkpts2_12[valid_indices_12]
        
        return filtered_mkpts1, filtered_mkpts2

    def _grid_based_anms(self, mkpts0: np.ndarray, mkpts1: np.ndarray, confidence: np.ndarray, grid_size: Tuple[int, int] = (8, 8), top_k_per_cell: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Subdivides the image into a grid and keeps the top_k matches per cell."""
        w, h = self.config['model']['target_res']
        cell_w = w / grid_size[0]
        cell_h = h / grid_size[1]
        
        grid_dictionary: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        
        for idx, (pt, conf) in enumerate(zip(mkpts0, confidence)):
            cell_x = int(pt[0] // cell_w)
            cell_y = int(pt[1] // cell_h)
            cell_coord = (cell_x, cell_y)
            
            if cell_coord not in grid_dictionary:
                grid_dictionary[cell_coord] = []
            grid_dictionary[cell_coord].append((idx, conf))
        
        filtered_indices: List[int] = []
        for cell_coord, matches in grid_dictionary.items():
            matches.sort(key=lambda x: x[1], reverse=True) # descending order
            filtered_indices.extend([match[0] for match in matches[:top_k_per_cell]])
        
        filtered_indices.sort(key=lambda idx: confidence[idx], reverse=True)
        
        filtered_mkpts0 = mkpts0[filtered_indices]
        filtered_mkpts1 = mkpts1[filtered_indices]
        
        return filtered_mkpts0, filtered_mkpts1
    
    def ransac(self, mkpts0: np.ndarray, mkpts1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Baseline: Estimates Essential Matrix using standard RANSAC """
        if len(mkpts0) < 5:
            print("Warning: Not enough points to estimate pose.")
            return np.eye(3), np.zeros((3, 1))

        K_scaled = self.loader.K_rescaled
        
        E, mask = cv2.findEssentialMat(
            mkpts0, 
            mkpts1, 
            cameraMatrix=K_scaled, 
            method=cv2.RANSAC,
            prob=self.config['pose_estimation']['ransac_prob'], 
            threshold=self.config['pose_estimation']['ransac_threshold'] 
        )
        
        if E is None or E.shape != (3, 3):
            print("Warning: Essential matrix estimation failed.")
            return np.eye(3), np.zeros((3, 1))
        
        inliers, R, t, mask_pose = cv2.recoverPose(E, mkpts0, mkpts1, cameraMatrix=K_scaled, mask=mask)
        return R, t

    def prosac(self, mkpts0: np.ndarray, mkpts1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Estimates Essential Matrix and recovers pose using OpenCV's USAC_PROSAC """
        if len(mkpts0) < 5:
            print("Not enough points to estimate pose.")
            return np.eye(3), np.zeros((3, 1))

        K_scaled = self.loader.K_rescaled
        
        E, mask = cv2.findEssentialMat(
            mkpts0, 
            mkpts1, 
            cameraMatrix=K_scaled, 
            method=cv2.USAC_PROSAC, 
            prob=self.config['pose_estimation']['ransac_prob'], 
            threshold=self.config['pose_estimation']['ransac_threshold']
        )
        
        if E is None:
            return np.eye(3), np.zeros((3, 1))
        
        inliers, R, t, mask_pose = cv2.recoverPose(E, mkpts0, mkpts1, cameraMatrix=K_scaled, mask=mask)

        if inliers < 10:
            print(f"Warning: Low inlier count ({inliers}) during pose recovery. Pose may be corrupted.")
        
        return R, t

    def estimate_pose(self, img_path_1: Path, img_path_2: Path) -> Tuple[np.ndarray, np.ndarray]:
        """ Estimates 6DOF pose from epipolar geometry """
        mkpts0, mkpts1, confidence = self._get_correspondences_img_pair(img_path_1, img_path_2)
        if self.method == 'vanilla':
            R, t = self.ransac(mkpts0, mkpts1)
        else:
            mkpts0_filtered, mkpts1_filtered = self._grid_based_anms(mkpts0, mkpts1, confidence)
            R, t = self.prosac(mkpts0_filtered, mkpts1_filtered)
        
        return R, t
    
    def estimate_pose_sliding_window(self, img_path_1: Path, img_path_2: Path, img_path_3: Path) -> Tuple[np.ndarray, np.ndarray]:
        """ Estimates 6DOF pose from consistent filtered feature matches over time """
        mkpts0_filtered, mkpts1_filtered = self._get_filtered_correspondences_3_frames(img_path_1, img_path_2, img_path_3)

        R, t = self.ransac(mkpts0_filtered, mkpts1_filtered)

        return R, t
    
    def _triangulate_points(self, R: np.ndarray, t: np.ndarray, mkpts1: np.ndarray, mkpts2: np.ndarray) -> np.ndarray:
        """Triangulates 3D points from 2D matches and relative pose."""
        K = self.loader.K_rescaled
        
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        
        # Homogeneous 4D coordinates
        pts4D = cv2.triangulatePoints(P1, P2, mkpts1.T, mkpts2.T)

        # 3D Cartesian coordinates
        pts3D = pts4D[:3, :] / pts4D[3, :]
        return pts3D.T

    def estimate_pose_pnp(self, img_path_1: Path, img_path_2: Path) -> Tuple[np.ndarray, np.ndarray]:
        """ Epipolar Rotation + PnP Scale """
        mkpts1, mkpts2, _ = self._get_correspondences_img_pair(img_path_1, img_path_2)
        
        R_est, t_est = self.ransac(mkpts1, mkpts2)
        
        if self.prev_3d_points is None:
            self.prev_3d_points = self._triangulate_points(R_est, t_est, mkpts1, mkpts2)
            self.prev_2d_points_f2 = mkpts2
            self.prev_pose_R = R_est
            self.prev_pose_t = t_est
            return R_est, t_est

        tree = KDTree(self.prev_2d_points_f2)
        distances, indices = tree.query(mkpts1, distance_upper_bound=self.config['feature_matching']['kdtree_threshold'])
        
        valid_2d_in_f2 = []
        valid_3d_in_f0 = []
        
        for i, (dist, j) in enumerate(zip(distances, indices)):
            if dist != float('inf'):
                valid_2d_in_f2.append(mkpts2[i]) 
                valid_3d_in_f0.append(self.prev_3d_points[j]) 
                
        valid_2d_in_f2 = np.ascontiguousarray(valid_2d_in_f2, dtype=np.float32)
        valid_3d_in_f0 = np.ascontiguousarray(valid_3d_in_f0, dtype=np.float32)
        
        R_rel = R_est
        t_rel = t_est
        
        # Run PnP to extract scale
        if len(valid_3d_in_f0) >= 6:
            R_02_guess = R_est @ self.prev_pose_R
            t_02_guess = R_est @ self.prev_pose_t + t_est
            rvec_guess, _ = cv2.Rodrigues(R_02_guess)
            
            pnp_flag = getattr(cv2, self.config['pose_estimation']['pnp_flags'])

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                valid_3d_in_f0, valid_2d_in_f2, self.loader.K_rescaled, distCoeffs=None, 
                flags=pnp_flag, 
                useExtrinsicGuess=True, 
                rvec=np.float64(rvec_guess), 
                tvec=np.float64(t_02_guess), 
                reprojectionError = self.config['pose_estimation']['pnp_reprojection_error']
            )
            
            if success:
                R_pnp, _ = cv2.Rodrigues(rvec)
                R_01 = self.prev_pose_R
                t_01 = self.prev_pose_t
                
                R_rel_pnp = R_pnp @ R_01.T
                t_rel_pnp = tvec - (R_rel_pnp @ t_01)
                
                # Extract the scalar magnitude from PnP
                scale = np.linalg.norm(t_rel_pnp)
                
                # Check the scale hasn't mathematically exploded
                if 0.05 < scale < 5.0:
                    # PnP's scale to the Essential Matrix's direction vector
                    t_rel = t_est * scale
                else:
                    print(f"Warning: PnP scale {scale:.2f} rejected. Falling back to unit scale.")
                    
        self.prev_3d_points = self._triangulate_points(R_rel, t_rel, mkpts1, mkpts2)
        self.prev_2d_points_f2 = mkpts2
        self.prev_pose_R = R_rel
        self.prev_pose_t = t_rel
        
        return R_rel, t_rel
