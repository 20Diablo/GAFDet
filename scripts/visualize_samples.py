import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
# from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud 
# No longer strictly necessary if we read bin directly

def main():
    parser = argparse.ArgumentParser(description="Visualize exported LiDAR/Radar samples individually")
    parser.add_argument('--input_dir', type=str, default='exported_samples', help='Directory containing exported samples')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    # Find sample folders with numeric sort
    try:
        sample_folders = sorted([f for f in os.listdir(args.input_dir) if f.startswith('sample_')], 
                                key=lambda x: int(x.split('_')[-1]))
    except:
        sample_folders = sorted([f for f in os.listdir(args.input_dir) if f.startswith('sample_')])
    
    if not sample_folders:
        print(f"No sample folders found in {args.input_dir}")
        return

    print(f"Found {len(sample_folders)} samples.")

    for sample_folder in sample_folders:
        sample_path = os.path.join(args.input_dir, sample_folder)
        lidar_dir = os.path.join(sample_path, 'lidar')
        radar_dir = os.path.join(sample_path, 'radar')
        vis_dir = os.path.join(sample_path, 'vis')
        
        os.makedirs(vis_dir, exist_ok=True)
        print(f"\nProcessing {sample_folder}...")

        # 1. Visualize LiDAR
        if os.path.exists(lidar_dir):
            lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
            for f in lidar_files:
                file_path = os.path.join(lidar_dir, f)
                vis_path = os.path.join(vis_dir, f"LIDAR_{f}.png")
                
                try:
                    # LiDAR in nuScenes/KITTI is usually x, y, z, intensity (float32)
                    scan = np.fromfile(file_path, dtype=np.float32)
                    # Reshape: dims usually 5 or 4
                    # Safe check for reshaping
                    num_points = scan.shape[0]
                    # Try reshaping to 5 based on mmdet3d convention, if fails try 4
                    if num_points % 5 == 0:
                        points = scan.reshape((-1, 5))[:, :4]
                    elif num_points % 4 == 0:
                        points = scan.reshape((-1, 4))[:, :4] # x,y,z,i
                    else:
                        print(f"Unknown LiDAR dimension for {f} (len={num_points})")
                        continue
                    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.scatter(points[:, 0], points[:, 1], s=0.1, c='blue')
                    ax.set_title(f"LiDAR: {f}")
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                    ax.set_xlim(-60, 60)
                    ax.set_ylim(-60, 60)
                    ax.grid(False) # Grid disabled as requested
                    plt.savefig(vis_path)
                    plt.close()
                    print(f"Saved LiDAR vis: {vis_path}")
                except Exception as e:
                    print(f"Error visualizing LiDAR {f}: {e}")

        # 2. Visualize Radar
        if os.path.exists(radar_dir):
            radar_files = [f for f in os.listdir(radar_dir) if f.endswith('.bin')]
            for f in radar_files:
                file_path = os.path.join(radar_dir, f)
                vis_path = os.path.join(vis_dir, f"RADAR_{f}.png")
                
                try:
                    # Fused Radar format: x, y, z, rcs, vx_comp, vy_comp, sweep_idx (7 fields)
                    scan = np.fromfile(file_path, dtype=np.float32)
                    num_points = scan.shape[0]
                    
                    if num_points % 7 == 0:
                        points = scan.reshape((-1, 7))
                    elif num_points % 6 == 0:
                        points = scan.reshape((-1, 6))
                    else:
                        print(f"Unknown Radar dimension for {f} (len={num_points}). Expected 7.")
                        continue
                    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    # Radar points: x is [0], y is [1]
                    # Color by RCS (index 3)
                    rcs = points[:, 3]
                    
                    sc = ax.scatter(points[:, 0], points[:, 1], s=3, c='red', marker='o')
                    # plt.colorbar(sc, label='RCS')
                    
                    ax.set_title(f"Fused Radar: {f}")
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                    # Radar range is typically larger, but let's keep consistent view or auto
                    ax.set_xlim(-60, 60)
                    ax.set_ylim(-60, 60)
                    ax.grid(False) # Grid disabled
                    plt.savefig(vis_path)
                    plt.close()
                    print(f"Saved Radar vis: {vis_path}")
                except Exception as e:
                    print(f"Error visualizing Radar {f}: {e}")

    print("\nVisualization complete!")

if __name__ == '__main__':
    main()
