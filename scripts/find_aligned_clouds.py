from nuscenes.nuscenes import NuScenes
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Find aligned LiDAR and Radar files in nuScenes")
    parser.add_argument('--dataroot', type=str, default='data/nuscenes', help='Path to nuScenes dataset root (e.g. d:/data/nuscenes)')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version (e.g. v1.0-mini, v1.0-trainval)')
    args = parser.parse_args()

    if not os.path.exists(args.dataroot):
        print(f"Error: Dataroot '{args.dataroot}' does not exist.")
        print("Please specify the correct path using --dataroot")
        return

    print(f"Loading nuScenes from {args.dataroot} ({args.version})...")
    try:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    except Exception as e:
        print(f"Failed to load nuScenes: {e}")
        return

    print(f"Total samples: {len(nusc.sample)}")
    
    # Iterate over samples to demonstrate alignment
    # In nuScenes, synchronization is handled by the 'sample' concept.
    # A sample is a keyframe ~2Hz where sensors are aligned (approx).
    
    print("\nDisplaying first 3 samples alignment:")
    for i, sample in enumerate(nusc.sample):
        print(f"\n=== Sample {i} ===")
        print(f"Token: {sample['token']}")
        print(f"Timestamp: {sample['timestamp']}")
        
        # 1. Get LiDAR info
        if 'LIDAR_TOP' in sample['data']:
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path = nusc.get_sample_data_path(lidar_token)
            print(f"[LiDAR] LIDAR_TOP: {lidar_path}")
        else:
            print("[LiDAR] LIDAR_TOP: Not found")
        
        # 2. Get Radar info
        # nuScenes has 5 radars
        radar_sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        for radar in radar_sensors:
            if radar in sample['data']:
                radar_token = sample['data'][radar]
                radar_path = nusc.get_sample_data_path(radar_token)
                print(f"[Radar] {radar}: {radar_path}")
            else:
                print(f"[Radar] {radar}: Not found")
                
        if i >= 2: 
            print("\n... stopping after 3 samples for demonstration.")
            break 

if __name__ == '__main__':
    main()
