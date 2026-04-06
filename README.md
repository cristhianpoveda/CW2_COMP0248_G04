# COMP0248 Coursework 2: Task 2
Pose estimation using LoFTR, depth estimation using Depth-Anything-V2, and image reprojection by fusing both.

### 1) How to run the code and generated outputs

**Environment Setup**
Run from the root directory
```bash
conda env create -f environment.yml
conda activate comp0248_cw2
```

**Running the Code**
1. **Data Preparation:** Extract data from rosbags, calibrate the camera and get the GT.
   ```bash
   python -m data_extraction.info_extraction
   python -m data_extraction.join_calibration_seq
   python -m data_extraction.calibration
   python -m data_extraction.get_seq
   python -m data_extraction.convert_gt --sequence Sequence_A
   ```
2. **Pose Estimation & Reprojection:**
   ```bash
   # Full evaluation (creates synthetic images and photometric stats)
   python -m image_reprojection.reproject_images --sequence_path data/Sequence_A/camera_color_image_raw
   
   # Trajectory-only generation (no reprojection)
   python -m image_reprojection.reproject_images --sequence_path data/Sequence_A/camera_color_image_raw --dont_reproject
   ```
3. **Trajectory Evaluation (EVO):**
   ```bash
   ./run_evo_eval.sh
   ```

**Outputs**
All outputs are saved within the `results/<Sequence_Name>/` directory:
* **`trajectories/`**: Contains the raw estimated trajectory files (`.tum` format).
* **`<run_name>/`**: Contains the `photometric_error_stats.txt`, `hyperparameters.yaml`, and a `synthetics_imgs/` folder with the rendered reprojected images.
* **`evo_eval/`**: Contains the `all_metrics_log.txt` (listing APE and RPE statistics) and the visual plot images (`.png`) aligned with Sim(3) scaling.

---

### 2) External code sources and own implementations

**External Code Sources and Libraries Used:**
* **LoFTR:** Pre-trained weights: `kornia` library.
* **Depth-Anything-V2:** Pre-trained weights: HuggingFace `transformers` pipeline.
* **OpenCV (`cv2`):** `cv2.findEssentialMat`, `cv2.recoverPose`, `cv2.solvePnPRansac`, and camera calibration functions.
* **EVO:** Evaluation tool for trajectory metrics (APE/RPE).
* **Rosbags:** `rosbags.highlevel.AnyReader`.

**Code Written from Scratch / Modified:**
* **Pipeline:** `PoseEstimation`, `DepthEstimator`, and `TensorLoader` were designed and coded from scratch to integrate the external models.
* **Photometric Reprojection:** The 3D point cloud triangulation, depth scaling, filtering, and pixel-splatting logic in `compute_photometric_error` was coded from scratch.
* **Data Extraction:** All scripts within the `data_extraction/` were written for this dataset.
* **Evaluation Scripts:** The `run_evo_eval.sh` script and the trajectory formatting logic were created from scratch.