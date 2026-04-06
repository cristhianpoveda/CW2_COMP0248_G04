import torch
import cv2
import numpy as np
import kornia
from kornia.feature import LoFTR
from pathlib import Path
from scipy.spatial import transform
import argparse
from scipy.spatial import KDTree
import yaml
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

def compute_photometric_error(img_path_1: Path, img_path_2: Path, depth1: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, target_res: Tuple[int, int], mkpts1: np.ndarray, mkpts2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Synthesizes Image 2 by warping Image 1 using the estimated pose and depth,
    then calculates the Mean Absolute Error (MAE) between the synthetic and real Image 2.
    """
    img1_bgr = cv2.resize(cv2.imread(str(img_path_1)), target_res)
    img2_bgr = cv2.resize(cv2.imread(str(img_path_2)), target_res)
    
    depth1 = cv2.resize(depth1, target_res, interpolation=cv2.INTER_NEAREST)

    h, w = img1_bgr.shape[:2]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    z = depth1.flatten()
    
    # Filter out invalid depth pixels (<= 0)
    valid_depth = z > 0
    u, v, z = u[valid_depth], v[valid_depth], z[valid_depth]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Image 1 pixels into 3D space
    X1 = (u - cx) * z / fx
    Y1 = (v - cy) * z / fy
    Z1 = z
    P1 = np.vstack((X1, Y1, Z1))

    t_mag_raw = np.linalg.norm(t)
    if t_mag_raw > 0:

        # Features shift between images
        pixel_shifts = np.linalg.norm(mkpts1 - mkpts2, axis=1)
        median_pixel_shift = np.median(pixel_shifts)
        
        # Scale: t = (pixel_shift * Z) / focal_length
        calculated_t_mag = (median_pixel_shift * np.median(Z1)) / fx
        
        t = (t / t_mag_raw) * calculated_t_mag
    
    t = t.reshape(3, 1)
    
    # Move points to cam 2 pose
    t = t.reshape(3, 1)
    P2 = R @ P1 + t
    
    X2, Y2, Z2 = P2[0, :], P2[1, :], P2[2, :]
    
    # Filter points behind Camera 2
    front_mask = Z2 > 0
    X2, Y2, Z2 = X2[front_mask], Y2[front_mask], Z2[front_mask]
    u_orig, v_orig = u[front_mask], v[front_mask]
    
    # Reproject 3D points onto Camera 2
    u2 = np.round((X2 * fx / Z2) + cx).astype(int)
    v2 = np.round((Y2 * fy / Z2) + cy).astype(int)
    
    # Filter points outside image 2
    bounds_mask = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)
    u2, v2 = u2[bounds_mask], v2[bounds_mask]
    u_orig, v_orig = u_orig[bounds_mask], v_orig[bounds_mask]
    Z2_final = Z2[bounds_mask]
    
    # Choose closer point if 2 pixels overlap
    sort_idx = np.argsort(Z2_final)[::-1]
    u2, v2 = u2[sort_idx], v2[sort_idx]
    u_orig, v_orig = u_orig[sort_idx], v_orig[sort_idx]
    
    synth_img2 = np.zeros_like(img2_bgr)
    
    synth_img2[v2, u2] = img1_bgr[v_orig, u_orig]
    
    # Photometric Error
    mask = np.zeros((h, w), dtype=bool)
    mask[v2, u2] = True

    if np.sum(mask) == 0:
        print("Warning: All points projected out of bounds. Returning NaN.")
        return np.nan, synth_img2
    
    img2_float = img2_bgr.astype(np.float32)
    synth_float = synth_img2.astype(np.float32)
    
    absolute_diff = np.abs(img2_float[mask] - synth_float[mask])
    photometric_error = np.mean(absolute_diff)
    
    return photometric_error, synth_img2 
    
def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix to a quaternion (qx, qy, qz, qw) natively using OpenCV."""
    r = transform.Rotation.from_matrix(matrix)
    return r.as_quat()

def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise 6DOF Pose Estimation using LoFTR and PROSAC.")
    parser.add_argument('--calib_file', type=str, default='calibration_data.npz', help='Path to the camera calibation .npz file')
    parser.add_argument('--sequence_path', type=str, default='Sequence_A/camera_color_image_raw', help='Path to the RGB sequence directory')
    parser.add_argument('--output', type=str, default='loftr_trajectory_A.tum', help='Name of the putput TUM trajectory file')
    parser.add_argument('--method', type=str, choices=['vanilla', 'prosac'], default='prosac', help='Pose estimation pipeline: vanilla (RANSAC), prosac (ANMS+PROSAC)')
    parser.add_argument('--correspondences', type=str, choices=['sliding_window', 'pairwise', 'pnp'], default='pnp', help='Matching strategy')
    parser.add_argument('--dont_reproject', action='store_true', help='Reproject image and calculate photometric error')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML hyperparameters file')
    
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded hyperparameters from {args.config}")
    except FileNotFoundError:
        print(f"Error: Config file {args.config} not found.")
        return
    
    try:
        calib_data = np.load(args.calib_file)
        try:
            K = calib_data['camera_matrix.npy']
            D = calib_data['dist_coeffs.npy']
        except KeyError:
            K = calib_data['camera_matrix']
            D = calib_data['dist_coeffs']
        print(f"Successfully loaded calibration data from {args.calib_file}")
    except Exception as e:
        print(f"Failed to load calibration data: {e}")
        return
    
    seq_dir = Path(args.sequence_path)
    if not seq_dir.exists():
        print(f"Error: Sequence directory {seq_dir} does not exist.")
        return

    images = sorted(list(seq_dir.glob("*.png")))
    if not images:
        print(f"Error: No PNG images found in {seq_dir}")
        return
    
    seq_name = seq_dir.parent.name

    pose_estimator = PoseEstimation(K, D, method=args.method, config=config)
    
    current_R = np.eye(3)
    current_t = np.zeros((3, 1))
    
    tum_trajectory: List[str] = []
    
    timestamp_first = images[0].stem.split('_')[1] 
    tum_trajectory.append(f"{timestamp_first} 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")

    if not args.dont_reproject:
        results_dir = Path(f"results/{seq_name}")
        synth_dir = results_dir / "synthetics_imgs"
        synth_dir.mkdir(parents=True, exist_ok=True)
        photometric_errors: List[float] = []

    print(f"Processing {len(images)} images from {seq_dir}...")
    
    for i in range(len(images) - 1):
        img_path_1 = images[i]
        img_path_2 = images[i+1]
        
        timestamp_ns = img_path_2.stem.split('_')[1]

        if args.correspondences == 'sliding_window' and (i + 2) < len(images):
            img_path_3 = images[i+2]
            R_rel, t_rel = pose_estimator.estimate_pose_sliding_window(img_path_1, img_path_2, img_path_3)

        elif args.correspondences == 'pnp':
            R_rel, t_rel = pose_estimator.estimate_pose_pnp(img_path_1, img_path_2)

        else:
            R_rel, t_rel = pose_estimator.estimate_pose(img_path_1, img_path_2)

        # Inverted transformation to get cam 2 in cam 1 frame
        R_cam = R_rel.T
        t_cam = -R_cam @ t_rel
        
        current_t = current_t + current_R @ t_cam
        current_R = current_R @ R_cam
        
        qx, qy, qz, qw = quaternion_from_matrix(current_R)
        tx, ty, tz = current_t.flatten()
        
        tum_line = f"{timestamp_ns} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n"
        tum_trajectory.append(tum_line)

        if not args.dont_reproject:
            depth_map_1 = pose_estimator._get_frame_data(img_path_1, require_depth=True)['depth_map']

            target_res = tuple(config['model']['target_res'])

            mkpts1, mkpts2, _ = pose_estimator._get_correspondences_img_pair(img_path_1, img_path_2)
            
            error, synthetic_img = compute_photometric_error(
                img_path_1, img_path_2, depth_map_1,
                pose_estimator.loader.K_rescaled, 
                R_rel, t_rel,
                target_res,
                mkpts1, mkpts2
            )
            
            photometric_errors.append(error)
            
            synth_filename = synth_dir / f"synth_{img_path_2.name}"
            cv2.imwrite(str(synth_filename), synthetic_img)
        
        print(f"Processed pair {i} -> {i+1}")

    if not args.dont_reproject and photometric_errors:
        valid_errors = [e for e in photometric_errors if not np.isnan(e)]
        if valid_errors:
            stats_file = results_dir / "photometric_error_stats.txt"
            with open(stats_file, "w") as f:
                f.write(f"--- Photometric Error Statistics ---\n")
                f.write(f"Sequence: {args.sequence_path}\n")
                f.write(f"Frames Evaluated: {len(photometric_errors)}\n")
                f.write(f"Mean Error (MAE): {np.mean(photometric_errors):.4f}\n")
                f.write(f"Median Error:     {np.median(photometric_errors):.4f}\n")
                f.write(f"Min Error:        {np.min(photometric_errors):.4f}\n")
                f.write(f"Max Error:        {np.max(photometric_errors):.4f}\n")
                f.write(f"Std Deviation:    {np.std(photometric_errors):.4f}\n")
            
            print(f"Photometric statistics saved to {stats_file}")

    with open(args.output, "w") as f:
        f.writelines(tum_trajectory)
    
    print(f"Trajectory saved to {args.output}")

if __name__ == '__main__':
    main()