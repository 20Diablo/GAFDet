import os
import shutil
import argparse
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

def main():
    parser = argparse.ArgumentParser(description="Export and visualize aligned LiDAR/Radar samples")
    parser.add_argument('--dataroot', type=str, default='data/nuscenes', help='Path to nuScenes dataset root')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version')
    parser.add_argument('--output_dir', type=str, default='exported_samples', help='Directory to save exported files')
    args = parser.parse_args()

    if not os.path.exists(args.dataroot):
        print(f"Error: Dataroot '{args.dataroot}' does not exist.")
        return

    print(f"Loading nuScenes from {args.dataroot} ({args.version})...")
    try:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    except Exception as e:
        print(f"Failed to load nuScenes: {e}")
        return

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    radar_sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

    print(f"\nProcessing first 5 samples...")
    for i, sample in enumerate(nusc.sample):
        if i >= 5:
            break
            
        print(f"\n--- Processing Sample {i} ---")
        sample_token = sample['token']
        sample_dir = os.path.join(args.output_dir, f'sample_{i}')
        lidar_dir = os.path.join(sample_dir, 'lidar')
        radar_dir = os.path.join(sample_dir, 'radar')
        
        os.makedirs(lidar_dir, exist_ok=True)
        os.makedirs(radar_dir, exist_ok=True)

        # Initialize visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.title(f"Sample {i}: LiDAR (Blue) vs Radar (Red)")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.grid(True)
        
        # 1. Process LiDAR
        if 'LIDAR_TOP' in sample['data']:
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path = nusc.get_sample_data_path(lidar_token)
            
            # Copy file
            dest_lidar_path = os.path.join(lidar_dir, os.path.basename(lidar_path))
            try:
                shutil.copy2(lidar_path, dest_lidar_path)
                print(f"Copied LiDAR: {os.path.basename(lidar_path)}")
            except IOError as e:
                print(f"Unable to copy LiDAR file: {e}")
            
            # Visualize LiDAR
            try:
                pc = LidarPointCloud.from_file(lidar_path)
                # Points are 4xN [x, y, z, intensity]
                # Filter for better visualization (optional, e.g. remove ground)
                # Plot x, y
                ax.scatter(pc.points[0, :], pc.points[1, :], s=0.1, c='blue', alpha=0.5, label='LiDAR')
            except Exception as e:
                print(f"Error reading LiDAR for vis: {e}")
        
        # 2. Process Radar
        all_radar_points = []
        for radar_name in radar_sensors:
            if radar_name in sample['data']:
                radar_token = sample['data'][radar_name]
                radar_path = nusc.get_sample_data_path(radar_token)
                
                # Copy file
                dest_radar_path = os.path.join(radar_dir, os.path.basename(radar_path))
                try:
                    shutil.copy2(radar_path, dest_radar_path)
                    print(f"Copied {radar_name}: {os.path.basename(radar_path)}")
                except IOError as e:
                    print(f"Unable to copy Radar file: {e}")

                # Visualize Radar
                try:
                    # Radar points need to be transformed to the Ego frame (or Lidar frame) to match Lidar
                    # For simplicity, let's transform everything to the EGO frame (which LIDAR_TOP is usually close to)
                    # OR transform Radar to Lidar frame directly if we want them perfectly aligned in the plot.
                    # Best practice: Output everything in LIDAR_TOP coordinates.
                    
                    # Get calibration for this radar
                    sd_record_radar = nusc.get('sample_data', radar_token)
                    cs_record_radar = nusc.get('calibrated_sensor', sd_record_radar['calibrated_sensor_token'])
                    pose_record_radar = nusc.get('ego_pose', sd_record_radar['ego_pose_token'])
                    
                    # Get calibration for LiDAR (Reference Frame)
                    lidar_token_ref = sample['data']['LIDAR_TOP']
                    sd_record_lidar = nusc.get('sample_data', lidar_token_ref)
                    cs_record_lidar = nusc.get('calibrated_sensor', sd_record_lidar['calibrated_sensor_token'])
                    pose_record_lidar = nusc.get('ego_pose', sd_record_lidar['ego_pose_token'])

                    pc_radar = RadarPointCloud.from_file(radar_path)
                    
                    # Transform Radar to Ego
                    pc_radar.rotate(Quaternion(cs_record_radar['rotation']).rotation_matrix)
                    pc_radar.translate(np.array(cs_record_radar['translation']))
                    
                    # Transform Ego to Global
                    pc_radar.rotate(Quaternion(pose_record_radar['rotation']).rotation_matrix)
                    pc_radar.translate(np.array(pose_record_radar['translation']))
                    
                    # Transform Global to Ego (LiDAR time)
                    pc_radar.translate(-np.array(pose_record_lidar['translation']))
                    pc_radar.rotate(Quaternion(pose_record_lidar['rotation']).rotation_matrix.T)
                    
                    # Transform Ego to LiDAR
                    pc_radar.translate(-np.array(cs_record_lidar['translation']))
                    pc_radar.rotate(Quaternion(cs_record_lidar['rotation']).rotation_matrix.T)

                    all_radar_points.append(pc_radar.points)
                    
                except Exception as e:
                    print(f"Error processing {radar_name} for vis: {e}")

        if all_radar_points:
            # Combine all radar points
            combined_radar = np.hstack(all_radar_points)
            ax.scatter(combined_radar[0, :], combined_radar[1, :], s=3, c='red', marker='x', alpha=1.0, label='Radar')

        # Limit plot range to typical perception range
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.legend(loc='upper right')
        
        vis_path = os.path.join(sample_dir, f'visualization_sample_{i}.png')
        plt.savefig(vis_path)
        plt.close()
        print(f"Saved visualization to: {vis_path}")

    print("\nDone! Exported samples are in:", os.path.abspath(args.output_dir))

if __name__ == '__main__':
    main()
