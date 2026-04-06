import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def find_closest_timestamp(target_ts: int, available_timestamps: list) -> int:
    return min(available_timestamps, key=lambda x: abs(x - target_ts))

def main() -> None:
    base_dir = PROJECT_ROOT / "data/GROUP_04_object_01"
    rgb_dir = base_dir / "camera_color_image_raw"
    depth_dir = base_dir / "camera_aligned_depth_to_color_image_raw"
    csv_path = base_dir / "phasespace_rigids/trajectory_log.csv"
    
    if not rgb_dir.exists() or not depth_dir.exists() or not csv_path.exists():
        print(f"Error: Missing sources in {base_dir}")
        return

    rgb_images = sorted(rgb_dir.glob("*.png"), key=lambda x: int(x.stem))
    total_frames = len(rgb_images)
    
    depth_images = sorted(depth_dir.glob("*.png"), key=lambda x: int(x.stem))
    available_depth_ts = [int(p.stem) for p in depth_images]
    
    if total_frames < 50 or not available_depth_ts:
        print("Error: Not enough frames")
        return
    
    t_start = int(rgb_images[0].stem)
    t_end = int(rgb_images[-1].stem)
    duration = (t_end - t_start) / 1e9
    true_fps = total_frames / duration
    step_size = int(math.ceil(true_fps / 3.0))
    
    print(f"Total RGB frames available: {total_frames} ({true_fps:.2f} fps)")
    
    # Sequence A
    seq_A_indices = list(range(0, total_frames, step_size))
    if len(seq_A_indices) < 50:
        back_step = seq_A_indices[-2]
        while len(seq_A_indices) < 50:
            seq_A_indices.append(back_step)
            back_step -= step_size
    seq_A_indices = seq_A_indices[:50]
    
    # Sequence B
    offset = step_size // 2
    seq_B_indices = list(range(offset, total_frames, step_size))
    if len(seq_B_indices) < 50:
        back_step = seq_B_indices[-2] if len(seq_B_indices) > 1 else 0
        while len(seq_B_indices) < 50:
            seq_B_indices.append(back_step)
            back_step -= step_size
    seq_B_indices = seq_B_indices[:50]

    # PhaseSpace
    phasespace_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['rigid_id'] == '1': 
                ts = int(row['timestamp_ns'])
                if ts not in phasespace_data:
                    phasespace_data[ts] = []
                phasespace_data[ts].append(row)
            
    available_ps_timestamps = sorted(list(phasespace_data.keys()))

    def create_sequence(seq_name: str, indices: list):
        seq_dir = PROJECT_ROOT / f"data/{seq_name}"
        seq_rgb_dir = seq_dir / "camera_color_image_raw"
        seq_depth_dir = seq_dir / "camera_aligned_depth_to_color_image_raw"
        seq_ps_dir = seq_dir / "phasespace_rigids"
        
        seq_rgb_dir.mkdir(parents=True, exist_ok=True)
        seq_depth_dir.mkdir(parents=True, exist_ok=True)
        seq_ps_dir.mkdir(parents=True, exist_ok=True)
        
        seq_csv_path = seq_ps_dir / "trajectory_log.csv"
        with open(seq_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, idx in enumerate(indices):
                rgb_file = rgb_images[idx]
                img_timestamp = int(rgb_file.stem)
                
                closest_depth_ts = find_closest_timestamp(img_timestamp, available_depth_ts)
                depth_file = depth_dir / f"{closest_depth_ts}.png"
                
                new_name_rgb = f"{i:02d}_{img_timestamp}.png"
                new_name_depth = f"{i:02d}_{img_timestamp}.png" 
                
                target_rgb = seq_rgb_dir / new_name_rgb
                if not target_rgb.exists():
                    target_rgb.symlink_to(rgb_file.resolve())
                    
                target_depth = seq_depth_dir / new_name_depth
                if depth_file.exists() and not target_depth.exists():
                    target_depth.symlink_to(depth_file.resolve())
                
                closest_ps_ts = find_closest_timestamp(img_timestamp, available_ps_timestamps)
                for row in phasespace_data[closest_ps_ts]:
                    synced_row = row.copy()
                    synced_row['timestamp_ns'] = img_timestamp 
                    writer.writerow(synced_row)

    create_sequence("Sequence_A", seq_A_indices)
    create_sequence("Sequence_B", seq_B_indices)
    print(f"\nSequences generated inside {PROJECT_ROOT}/data/")

if __name__ == "__main__":
    main()