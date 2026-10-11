import os
import shutil
import argparse
import numpy as np
from functools import reduce
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

def main():
    parser = argparse.ArgumentParser(description="Export aligned LiDAR/Radar samples (Fused)")
    parser.add_argument('--dataroot', type=str, default='data/nuScenes', help='Path to nuScenes dataset root')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--output_dir', type=str, default='exported_samples', help='Directory to save exported files')
    args = parser.parse_args()

    if not os.path.exists(args.dataroot):
        print(f"Error: Dataroot '{args.dataroot}' does not exist.")
        print(f"Absolute path checked: {os.path.abspath(args.dataroot)}")
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
    
    # Fields to save: x, y, z, rcs, vx_comp, vy_comp, sweep_idx
    # We will use sweep_idx = 0 for all points since we are only using the current frame.
    SAVE_FIELDS = [0, 1, 2, 5, 8, 9, -1] 

    print(f"\nProcessing first 10 samples...")
    for i, sample in enumerate(nusc.sample):
        if i >= 10:
            break
            
        print(f"\n--- Processing Sample {i} ---")
        sample_dir = os.path.join(args.output_dir, f'sample_{i}')
        lidar_dir = os.path.join(sample_dir, 'lidar')
        radar_dir = os.path.join(sample_dir, 'radar')
        
        os.makedirs(lidar_dir, exist_ok=True)
        os.makedirs(radar_dir, exist_ok=True)
        
        # 1. Export LiDAR
        if 'LIDAR_TOP' in sample['data']:
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path = nusc.get_sample_data_path(lidar_token)
            lidar_data = nusc.get('sample_data', lidar_token)
            
            dest_lidar_path = os.path.join(lidar_dir, os.path.basename(lidar_path))
            try:
                shutil.copy2(lidar_path, dest_lidar_path)
                print(f"Copied LiDAR: {os.path.basename(lidar_path)}")
            except IOError as e:
                print(f"Unable to copy LiDAR file: {e}")
                continue

            # Prepare for Radar alignment (LiDAR Frame as Reference)
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample['data'][ref_chan]
            ref_sd_rec = nusc.get('sample_data', ref_sd_token)
            ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
            ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
            
            # Homogeneous transform from ego car frame to reference frame
            ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                            Quaternion(ref_cs_rec['rotation']), inverse=True)
            # Homogeneous transformation matrix from global to _current_ ego car frame
            car_from_global = transform_matrix(ref_pose_rec['translation'],
                                               Quaternion(ref_pose_rec['rotation']), inverse=True)
            
            # 2. Process and Fuse Radar
            all_radar_points = np.zeros((18, 0))
            all_pc = RadarPointCloud(all_radar_points) # Initialize empty point cloud container
            
            # Filter settings (similar to gen_radar_bev.py but simplified or standard)
            # Using standard filtering for valid points
            invalid_states = [0, 4, 8, 9, 10, 11, 12, 15, 16, 17]
            dynprop_states = list(range(8))
            ambig_states = [3]

            for radar_name in radar_sensors:
                if radar_name in sample['data']:
                    radar_token = sample['data'][radar_name]
                    current_sd_rec = nusc.get('sample_data', radar_token)
                    radar_path = nusc.get_sample_data_path(radar_token)
                    
                    # Load point cloud
                    current_pc = RadarPointCloud.from_file(radar_path, invalid_states, dynprop_states, ambig_states)
                    
                    # Get past pose (current radar pose)
                    current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                    global_from_car = transform_matrix(current_pose_rec['translation'],
                                                       Quaternion(current_pose_rec['rotation']), inverse=False)

                    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                    current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                    car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                        inverse=False)

                    # Fuse four transformation matrices into one and perform transform.
                    # Radar -> Ego -> Global -> RefEgo -> RefLiDAR
                    trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                    current_pc.transform(trans_matrix)
                    
                    # Rotate velocity to reference frame
                    velocities = current_pc.points[8:10, :]  # Compensated velocity (vx_comp, vy_comp)
                    velocities = np.vstack((velocities, np.zeros(current_pc.points.shape[1])))
                    # SensorFrame -> EgoFrame -> GlobalFrame -> RefEgoFrame -> RefSensorFrame(LiDAR) 
                    # Note: Velocities are vectors, not points. Translation doesn't apply.
                    # Rotation: R_sensor2ego @ v
                    velocities = np.dot(Quaternion(current_cs_rec['rotation']).rotation_matrix, velocities)
                    # Ego -> Global: R_ego2global @ v
                    velocities = np.dot(Quaternion(current_pose_rec['rotation']).rotation_matrix, velocities)
                    # Global -> RefEgo: R_global2refego @ v = R_refego2global.T @ v
                    velocities = np.dot(Quaternion(ref_pose_rec['rotation']).rotation_matrix.T, velocities)
                    # RefEgo -> RefLiDAR: R_refego2lidar @ v = R_lidar2refego.T @ v
                    velocities = np.dot(Quaternion(ref_cs_rec['rotation']).rotation_matrix.T, velocities)
                    
                    current_pc.points[8:10, :] = velocities[0:2, :]

                    # Save sweep index (0 for current frame)
                    current_pc.points[-1, :] = 0
                    
                    # Stack
                    all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Select fields and Transpose
            # gen_radar_bev.py: points = all_pc.points[SAVE_FIELDS, :].T.astype(np.float32)
            # SAVE_FIELDS = [0, 1, 2, 5, 8, 9, -1] -> x, y, z, rcs, vx_comp, vy_comp, sweep_idx
            fused_points = all_pc.points[SAVE_FIELDS, :].T.astype(np.float32)
            
            # Save to bin file
            # Use same name as lidar file but .bin extension (already is)
            radar_filename = os.path.basename(lidar_path)
            dest_radar_path = os.path.join(radar_dir, radar_filename)
            
            fused_points.tofile(dest_radar_path)
            print(f"Saved Fused Radar: {radar_filename} (Points: {fused_points.shape[0]})")

    print("\nDone! Exported samples are in:", os.path.abspath(args.output_dir))

if __name__ == '__main__':
    main()
