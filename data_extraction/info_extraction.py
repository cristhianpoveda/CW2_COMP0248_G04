import json
from pathlib import Path
import numpy as np
import cv2
from rosbags.highlevel import AnyReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    bags_dir = PROJECT_ROOT / 'data/rosbags'
    bag_files = list(bags_dir.glob('*.bag'))

    if not bag_files:
        print(f"No .bag files found in {bags_dir}.")
        return

    for bag_path in bag_files:
        bag_name = bag_path.stem
        
        output_dir = PROJECT_ROOT / f'data/{bag_name}'
        print(f"\n========== Processing {bag_path.name} ==========")
        
        saved_camera_info = set()

        with AnyReader([bag_path]) as bag:
            phasespace_csv_path = output_dir / "phasespace_rigids/trajectory_log.csv"
            phasespace_csv_path.parent.mkdir(parents=True, exist_ok=True)
            phasespace_file = None

            try:
                for connection, timestamp, rawdata in bag.messages():
                    clean_topic = connection.topic.strip('/').replace('/', '_')
                    topic_dir = output_dir / clean_topic
                    topic_dir.mkdir(parents=True, exist_ok=True)
                                    
                    try:
                        msg = bag.deserialize(rawdata, connection.msgtype)
                    except Exception as e:
                        print(f"Warning: Could not deserialize {connection.msgtype} on {connection.topic}: {e}")
                        continue

                    # Images (RGB and Aligned Depth)
                    if connection.msgtype in ['sensor_msgs/msg/Image', 'sensor_msgs/Image']:
                        try:
                            if msg.encoding in ['16UC1', 'mono16']:
                                img_array = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
                            else:
                                channels = 3 if msg.encoding in ['rgb8', 'bgr8'] else 1
                                img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
                                
                                if msg.encoding == 'rgb8':
                                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                    
                            img_filename = topic_dir / f"{timestamp}.png"
                            cv2.imwrite(str(img_filename), img_array)
                            
                        except Exception as e:
                            print(f"Error saving image at {timestamp}: {e}")

                    # Camera Info
                    elif connection.msgtype in ['sensor_msgs/msg/CameraInfo', 'sensor_msgs/CameraInfo']:
                        if clean_topic not in saved_camera_info:
                            info_dict = {
                                "width": msg.width,
                                "height": msg.height,
                                "distortion_model": msg.distortion_model,
                                "D": list(msg.D),
                                "K": list(msg.K),
                                "R": list(msg.R),
                                "P": list(msg.P)
                            }
                            
                            info_filename = topic_dir / "camera_info.json"
                            with open(info_filename, "w") as f:
                                json.dump(info_dict, f, indent=4)
                                
                            saved_camera_info.add(clean_topic)
                            print(f"Saved CameraInfo for {connection.topic}")
                    
                    # Custom PhaseSpace Rigids
                    elif connection.msgtype == 'phasespace_msgs/msg/Rigids':
                        if phasespace_file is None:
                            phasespace_file = open(phasespace_csv_path, "w")
                            phasespace_file.write("timestamp_ns,rigid_id,x,y,z,qw,qx,qy,qz,cond,flags\n")
                        
                        for rigid in msg.rigids:
                            row = f"{timestamp},{rigid.id},{rigid.x},{rigid.y},{rigid.z},{rigid.qw},{rigid.qx},{rigid.qy},{rigid.qz},{rigid.cond},{rigid.flags}\n"
                            phasespace_file.write(row)

                if phasespace_file is not None:
                    phasespace_file.close()
            
            except RuntimeError as e:
                if "LZ4F_decompress" in str(e):
                    print(f"\n[WARNING] Reached a corrupted LZ4 chunk in {bag_name}. Salvaged all data up to this point and moving to the next bag.")
                else:
                    raise e

    print("\nExtraction complete!")

if __name__ == '__main__':
    main()