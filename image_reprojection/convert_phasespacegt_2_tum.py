import csv
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def convert_csv_to_tum(csv_path: Path, tum_path: Path) -> None:
    
    tum_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'r') as f_in, open(tum_path, 'w') as f_out:
        reader = csv.DictReader(f_in)
        count = 0
        for row in reader:
            ts = row['timestamp_ns']
            x, y, z = row['x'], row['y'], row['z']
            
            qx, qy, qz, qw = row['qx'], row['qy'], row['qz'], row['qw']
            
            f_out.write(f"{ts} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
            count += 1
            
    print(f"Converted {count} rows. Ground truth saved to {tum_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ground truth CSV to TUM format.")
    
    parser.add_argument('--sequence', type=str, default='Sequence_A', help='Name of the sequence folder (e.g., Sequence_A)')
    
    args = parser.parse_args()
    
    csv_file = PROJECT_ROOT / f"data/{args.sequence}/phasespace_rigids/trajectory_log.csv"
    
    tum_file = PROJECT_ROOT / f"results/{args.sequence}/groundtruth.tum"
    
    if not csv_file.exists():
        print(f"Error: Could not find the CSV file at {csv_file}")
    else:
        convert_csv_to_tum(csv_file, tum_file)

if __name__ == "__main__":
    main()