import csv

def convert_csv_to_tum(csv_path, tum_path):
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

if __name__ == "__main__":
    # Point this to your sequence's CSV
    csv_file = "Sequence_B/phasespace_rigids/trajectory_log.csv"
    tum_file = "groundtruth_B.tum"
    
    convert_csv_to_tum(csv_file, tum_file)