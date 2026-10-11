import os
import shutil
import argparse
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Export and Visualize Scene Data (Day, Night, Rain)")
    parser.add_argument('--dataroot', type=str, default='data/nuScenes', help='Path to nuScenes dataset root')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--info_path', type=str, default='data/nuScenes/nuscenes_infos_val.pkl', help='Path to infos pkl')
    parser.add_argument('--output_dir', type=str, default='exported_scenes', help='Root directory for output')
    return parser.parse_args()

def export_lidar(nusc, sample, output_dir):
    if 'LIDAR_TOP' in sample['data']:
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path = nusc.get_sample_data_path(lidar_token)
        dest_path = os.path.join(output_dir, os.path.basename(lidar_path))
        try:
            shutil.copy2(lidar_path, dest_path)
            return dest_path
        except IOError as e:
            print(f"  Error copying LiDAR: {e}")
            return None
    return None

def process_and_fuse_radar(nusc, sample, output_dir, radar_sensors):
    # Setup for fusion aligned to LIDAR_TOP
    ref_chan = 'LIDAR_TOP'
    ref_sd_token = sample['data'][ref_chan]
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    
    ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                    Quaternion(ref_cs_rec['rotation']), inverse=True)
    car_from_global = transform_matrix(ref_pose_rec['translation'],
                                       Quaternion(ref_pose_rec['rotation']), inverse=True)
    
    all_radar_points = np.zeros((18, 0))
    all_pc = RadarPointCloud(all_radar_points)
    
    invalid_states = [0, 4, 8, 9, 10, 11, 12, 15, 16, 17]
    dynprop_states = list(range(8))
    ambig_states = [3]

    for radar_name in radar_sensors:
        if radar_name in sample['data']:
            radar_token = sample['data'][radar_name]
            current_sd_rec = nusc.get('sample_data', radar_token)
            radar_path = nusc.get_sample_data_path(radar_token)
            
            current_pc = RadarPointCloud.from_file(radar_path, invalid_states, dynprop_states, ambig_states)
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)
            
            # Velocity compensation
            velocities = current_pc.points[8:10, :]
            velocities = np.vstack((velocities, np.zeros(current_pc.points.shape[1])))
            velocities = np.dot(Quaternion(current_cs_rec['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(current_pose_rec['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_pose_rec['rotation']).rotation_matrix.T, velocities)
            velocities = np.dot(Quaternion(ref_cs_rec['rotation']).rotation_matrix.T, velocities)
            
            current_pc.points[8:10, :] = velocities[0:2, :]
            current_pc.points[-1, :] = 0 # Sweep index 0
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

    SAVE_FIELDS = [0, 1, 2, 5, 8, 9, -1] # x, y, z, rcs, vx_comp, vy_comp, sweep_idx
    fused_points = all_pc.points[SAVE_FIELDS, :].T.astype(np.float32)
    
    # Save using LIDAR filename but .bin extension
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_path = nusc.get_sample_data_path(lidar_token)
    radar_filename = "RADAR_" + os.path.basename(lidar_path)
    dest_path = os.path.join(output_dir, radar_filename)
    
    fused_points.tofile(dest_path)
    return dest_path

def visualize_lidar(bin_path, save_path, title):
    try:
        scan = np.fromfile(bin_path, dtype=np.float32)
        num_points = scan.shape[0]
        if num_points % 5 == 0:
            points = scan.reshape((-1, 5))[:, :4]
        elif num_points % 4 == 0:
            points = scan.reshape((-1, 4))[:, :4]
        else:
            print(f"    [Vis Warn] Unknown LiDAR dim for {os.path.basename(bin_path)}")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(points[:, 0], points[:, 1], s=0.1, c='blue')
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"    [Vis Error] LiDAR: {e}")

def visualize_radar(bin_path, save_path, title):
    try:
        scan = np.fromfile(bin_path, dtype=np.float32)
        num_points = scan.shape[0]
        if num_points % 7 == 0:
            points = scan.reshape((-1, 7))
        elif num_points % 6 == 0:
            points = scan.reshape((-1, 6))
        else:
            print(f"    [Vis Warn] Unknown Radar dim for {os.path.basename(bin_path)}")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        # Color by RCS (index 3) is optional, user requested 'gray' in other script, let's stick to simple red/style from ref
        ax.scatter(points[:, 0], points[:, 1], s=3, c='red', marker='o')
        
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"    [Vis Error] Radar: {e}")

def main():
    args = parse_args()
    
    # Define Scenes and Indices
    # Note: Indices are inclusive [start, end]
    # Day: 1011-1016 (6 frames)
    # Night: 5682-5687 (6 frames)
    # Day_Rain: 4022-4027 (6 frames)
    # Night_Rain: 5811-5816 (6 frames)
    
    SCENES = {
        'Day': range(1011, 1017),
        'Night': range(5682, 5688),
        'Day and Rain': range(4022, 4028),
        'Night and Rain': range(5811, 5817)
    }
    
    RADAR_SENSORS = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

    print(f"Loading infos from {args.info_path}...")
    try:
        infos = mmcv.load(args.info_path)
    except Exception as e:
        print(f"Error loading infos: {e}")
        return

    print(f"Loading nuScenes from {args.dataroot}...")
    try:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    except Exception as e:
        print(f"Error loading nuScenes: {e}")
        return

    for category, indices in SCENES.items():
        print(f"\nProcessing Category: {category}")
        cat_dir = os.path.join(args.output_dir, category)
        
        for idx in indices:
            if idx >= len(infos):
                print(f"  [Warn] Index {idx} out of bounds (max {len(infos)-1}). Skipping.")
                continue

            # Creating subfolder for this index
            index_dir = os.path.join(cat_dir, f"index_{idx}")
            os.makedirs(index_dir, exist_ok=True)
            
            info = infos[idx]
            token = info['sample_token']
            sample = nusc.get('sample', token)
            
            print(f"  Exporting Index {idx} (Token: {token})...")
            
            # 1. Export LiDAR
            lidar_bin_path = export_lidar(nusc, sample, index_dir)
            
            # 2. Export Fused Radar
            radar_bin_path = process_and_fuse_radar(nusc, sample, index_dir, RADAR_SENSORS)
            
            # 3. Visualization
            if lidar_bin_path:
                vis_name = f"LIDAR_{os.path.basename(lidar_bin_path)}.png"
                visualize_lidar(lidar_bin_path, os.path.join(index_dir, vis_name), f"{category} - Index {idx} - LiDAR")
                
            if radar_bin_path:
                vis_name = f"RADAR_{os.path.basename(radar_bin_path)}.png"
                visualize_radar(radar_bin_path, os.path.join(index_dir, vis_name), f"{category} - Index {idx} - Radar")

    print(f"\nAll done! Results saved in '{args.output_dir}'")

if __name__ == '__main__':
    main()
