import cv2
import numpy as np
from pathlib import Path
from scipy.spatial import transform
import argparse
import yaml
from typing import Tuple, List
from datetime import datetime
from image_reprojection.pose_estimation import PoseEstimation

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_photometric_error(img_path_1: Path, img_path_2: Path, depth1: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, target_res: Tuple[int, int], mkpts1: np.ndarray, mkpts2: np.ndarray, no_splatting) -> Tuple[float, float, np.ndarray]:
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

    mask = np.zeros((h, w), dtype=bool)

    # Splatting: pixel dilation
    if not no_splatting:

        kernel = np.ones((2, 2), np.uint8) 
        
        synth_img2 = cv2.dilate(synth_img2, kernel, iterations=1)
        
        mask = (synth_img2[:, :, 0] > 0) | (synth_img2[:, :, 1] > 0) | (synth_img2[:, :, 2] > 0)
    
    else:
        
        mask[v2, u2] = True

    # Photometric Error
    if np.sum(mask) == 0:
        print("Warning: All points projected out of bounds. Returning NaN.")
        return np.nan, np.nan, synth_img2
    
    img2_float = img2_bgr.astype(np.float32)
    synth_float = synth_img2.astype(np.float32)
    
    absolute_diff = np.abs(img2_float[mask] - synth_float[mask])
    photometric_error_mae = np.mean(absolute_diff)
    
    squared_diff = np.square(img2_float[mask] - synth_float[mask])
    photometric_error_rmse = np.sqrt(np.mean(squared_diff))
    
    return photometric_error_mae, photometric_error_rmse, synth_img2 
    
def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix to a quaternion (qx, qy, qz, qw) natively using OpenCV."""
    r = transform.Rotation.from_matrix(matrix)
    return r.as_quat()

def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise 6DOF Pose Estimation using LoFTR and PROSAC.")
    parser.add_argument('--calib_file', type=str, default=str(PROJECT_ROOT / 'data/calibration_data.npz'), help='Path to the camera calibation .npz file')
    parser.add_argument('--sequence_path', type=str, default=str(PROJECT_ROOT / 'data/Sequence_A/camera_color_image_raw'), help='Path to the RGB sequence directory')
    parser.add_argument('--method', type=str, choices=['vanilla', 'prosac'], default='vanilla', help='Pose estimation pipeline: vanilla (RANSAC), prosac (ANMS+PROSAC)')
    parser.add_argument('--correspondences', type=str, choices=['sliding_window', 'pairwise', 'pnp'], default='pnp', help='Matching strategy')
    parser.add_argument('--dont_reproject', action='store_true', help='Reproject image and calculate photometric error')
    parser.add_argument('--no_splatting', action='store_true', help='Perform splatting to synthetic image')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'config/config.yaml'), help='Path to the YAML hyperparameters file')
    
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    splat_str = "nosplat" if getattr(args, 'no_splatting', False) else "splat"
    run_name = f"{args.method}_{args.correspondences}_{splat_str}_{timestamp}"
    seq_results_base = PROJECT_ROOT / f"results/{seq_name}"

    if not args.dont_reproject:
        
        results_dir = seq_results_base / run_name
        synth_dir = results_dir / "synthetics_imgs"
        synth_dir.mkdir(parents=True, exist_ok=True)
        
        hyperparameters_log = {
            'command_line_args': vars(args),
            'yaml_config': config
        }
        
        hyperparams_file = results_dir / "hyperparameters.yaml"
        with open(hyperparams_file, "w") as f:
            yaml.dump(hyperparameters_log, f, default_flow_style=False, sort_keys=False)
            
        print(f"Evaluation Run initialized. Hyperparameters saved to {hyperparams_file}")

        mae_errors: List[float] = []
        rmse_errors: List[float] = []
    else:
        
        results_dir = seq_results_base / "trajectories"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Trajectory-Only Run. Outputs will save to {results_dir}")

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
            
            mae, rmse, synthetic_img = compute_photometric_error(
                img_path_1, img_path_2, depth_map_1,
                pose_estimator.loader.K_rescaled, 
                R_rel, t_rel,
                target_res,
                mkpts1, mkpts2,
                args.no_splatting
            )
            
            mae_errors.append(mae)
            rmse_errors.append(rmse)
            
            synth_filename = synth_dir / f"synth_{img_path_2.name}"
            orig_h, orig_w = cv2.imread(str(img_path_1)).shape[:2]
            
            synthetic_img_display = cv2.resize(synthetic_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(synth_filename), synthetic_img_display)
        
        print(f"Processed pair {i} -> {i+1}")

    if not args.dont_reproject and mae_errors and rmse_errors:
        
        valid_mae = [e for e in mae_errors if not np.isnan(e)]
        valid_rmse = [e for e in rmse_errors if not np.isnan(e)]

        if valid_mae and valid_rmse:
            stats_file = results_dir / "photometric_error_stats.txt"
            with open(stats_file, "w") as f:
                f.write(f"--- Photometric Error Statistics ---\n")
                f.write(f"Sequence: {args.sequence_path}\n")
                f.write(f"Frames Evaluated: {len(valid_mae)} / {len(mae_errors)}\n\n")
                
                f.write(f"--- MAE (Mean Absolute Error) ---\n")
                f.write(f"Mean:     {np.mean(valid_mae):.4f}\n")
                f.write(f"Median:   {np.median(valid_mae):.4f}\n")
                f.write(f"Min:      {np.min(valid_mae):.4f}\n")
                f.write(f"Max:      {np.max(valid_mae):.4f}\n")
                f.write(f"Std Dev:  {np.std(valid_mae):.4f}\n\n")
                
                f.write(f"--- RMSE (Root Mean Square Error) ---\n")
                f.write(f"Mean:     {np.mean(valid_rmse):.4f}\n")
                f.write(f"Median:   {np.median(valid_rmse):.4f}\n")
                f.write(f"Min:      {np.min(valid_rmse):.4f}\n")
                f.write(f"Max:      {np.max(valid_rmse):.4f}\n")
                f.write(f"Std Dev:  {np.std(valid_rmse):.4f}\n")
            
            print(f"Photometric statistics saved to {stats_file}")

    trajectory_filename = f"traj_{run_name}.tum"
    trajectory_output_path = results_dir / trajectory_filename

    with open(trajectory_output_path, "w") as f:
        f.writelines(tum_trajectory)
    
    print(f"Trajectory saved to {trajectory_output_path}")

if __name__ == '__main__':
    main()