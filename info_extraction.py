import os
import json
from pathlib import Path
import numpy as np
import cv2
from rosbags.highlevel import AnyReader

def extract_bags():
    
    bag_files = list(Path('.').glob('*.bag'))

    if not bag_files:
        print("No .bag files found in the current directory.")
        return

    for bag_path in bag_files:
        bag_name = bag_path.stem
        print(f"\n========== Processing {bag_path.name} ==========")
        
        saved_camera_info = set()

        with AnyReader([bag_path]) as bag:
            
            phasespace_log_path = Path(bag_name) / "phasespace_rigids" / "trajectory_log.txt"
            phasespace_file = None

            try:
                
                for connection, timestamp, rawdata in bag.messages():
                    
                    clean_topic = connection.topic.strip('/').replace('/', '_')
                    topic_dir = Path(bag_name) / clean_topic
                    topic_dir.mkdir(parents=True, exist_ok=True)
                                    
                    # Deserialize the message
                    try:
                        msg = bag.deserialize(rawdata, connection.msgtype)
                    except Exception as e:
                        print(f"Warning: Could not deserialize {connection.msgtype} on {connection.topic}: {e}")
                        continue

                    # Images (RGB and Aligned Depth)
                    if connection.msgtype in ['sensor_msgs/msg/Image', 'sensor_msgs/Image']:
                        try:
                            # RealSense Depth (16-bit)
                            if msg.encoding in ['16UC1', 'mono16']:
                                img_array = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
                            # RealSense RGB (8-bit)
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
                                "D": list(msg.D), # Distortion coefficients
                                "K": list(msg.K), # Intrinsic matrix
                                "R": list(msg.R), # Rectification matrix
                                "P": list(msg.P)  # Projection/camera matrix
                            }
                            
                            info_filename = topic_dir / "camera_info.json"
                            with open(info_filename, "w") as f:
                                json.dump(info_dict, f, indent=4)
                                
                            saved_camera_info.add(clean_topic)
                            print(f"Saved CameraInfo for {connection.topic}")
                    
                    # Custom PhaseSpace Rigids (Export to CSV)

                    elif connection.msgtype == 'phasespace_msgs/msg/Rigids':
                        
                        phasespace_csv_path = Path(bag_name) / "phasespace_rigids" / "trajectory_log.csv"
                        
                        if phasespace_file is None:
                            phasespace_file = open(phasespace_csv_path, "w")
                            # Column headers
                            phasespace_file.write("timestamp_ns,rigid_id,x,y,z,qw,qx,qy,qz,cond,flags\n")
                        
                        for rigid in msg.rigids:
                            # rigid body's pose as a row in the CSV
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
    extract_bags()