import mmcv
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from radar_converter_coordinate_v1 import get_available_scenes

## from BEVcar
val_sun = \
        ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
         'scene-0035', 'scene-0036',
         'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095', 'scene-0096', 'scene-0097',
         'scene-0098', 'scene-0099',
         'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107',
         'scene-0108', 'scene-0109',
         'scene-0110', 'scene-0221', 'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273',
         'scene-0274', 'scene-0275',
         'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
         'scene-0345', 'scene-0346',
         'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524', 'scene-0552', 'scene-0553',
         'scene-0554', 'scene-0555',
         'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559', 'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563',
         'scene-0564', 'scene-0565',
         'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780', 'scene-0781', 'scene-0782',
         'scene-0783', 'scene-0784',
         'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797', 'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802',
         'scene-0916', 'scene-0917',
         'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924', 'scene-0925', 'scene-0926',
         'scene-0927', 'scene-0928',
         'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962', 'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968',
         'scene-0969', 'scene-0971',
         'scene-0972']
val_night = \
        ['scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065',
         'scene-1066', 'scene-1067', 'scene-1068',
         'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

val_rain = \
         ['scene-0625', 'scene-0626', 'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633',
          'scene-0634', 'scene-0635', 'scene-0636',
          'scene-0637', 'scene-0638', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907', 'scene-0908',
          'scene-0909', 'scene-0910', 'scene-0911',
          'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915']

def generate_info(nusc, scenes, max_cam_sweeps=6, max_lidar_sweeps=10):
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            cam_datas = list()
            lidar_datas = list()
            info['scene_name'] = nusc.get('scene', cur_scene['token'])['name']
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            lidar_names = ['LIDAR_TOP']
            cam_infos = dict()
            lidar_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get(
                    'ego_pose', cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)
                sweep_lidar_info = dict()
                sweep_lidar_info['sample_token'] = lidar_data['sample_token']
                sweep_lidar_info['ego_pose'] = nusc.get(
                    'ego_pose', lidar_data['ego_pose_token'])
                sweep_lidar_info['timestamp'] = lidar_data['timestamp']
                sweep_lidar_info['filename'] = lidar_data['filename']
                sweep_lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_infos[lidar_name] = sweep_lidar_info

            lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
            cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            for k, cam_data in enumerate(cam_datas):
                sweep_cam_data = cam_data
                for j in range(max_cam_sweeps):
                    if sweep_cam_data['prev'] == '':
                        break
                    else:
                        sweep_cam_data = nusc.get('sample_data',
                                                  sweep_cam_data['prev'])
                        sweep_cam_info = dict()
                        sweep_cam_info['sample_token'] = sweep_cam_data[
                            'sample_token']
                        if sweep_cam_info['sample_token'] != cam_data[
                                'sample_token']:
                            break
                        sweep_cam_info['ego_pose'] = nusc.get(
                            'ego_pose', cam_data['ego_pose_token'])
                        sweep_cam_info['timestamp'] = sweep_cam_data[
                            'timestamp']
                        sweep_cam_info['is_key_frame'] = sweep_cam_data[
                            'is_key_frame']
                        sweep_cam_info['height'] = sweep_cam_data['height']
                        sweep_cam_info['width'] = sweep_cam_data['width']
                        sweep_cam_info['filename'] = sweep_cam_data['filename']
                        sweep_cam_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        cam_sweeps[j][cam_names[k]] = sweep_cam_info

            for k, lidar_data in enumerate(lidar_datas):
                sweep_lidar_data = lidar_data
                for j in range(max_lidar_sweeps):
                    if sweep_lidar_data['prev'] == '':
                        break
                    else:
                        sweep_lidar_data = nusc.get('sample_data',
                                                    sweep_lidar_data['prev'])
                        sweep_lidar_info = dict()
                        sweep_lidar_info['sample_token'] = sweep_lidar_data[
                            'sample_token']
                        if sweep_lidar_info['sample_token'] != lidar_data[
                                'sample_token']:
                            break
                        sweep_lidar_info['ego_pose'] = nusc.get(
                            'ego_pose', sweep_lidar_data['ego_pose_token'])
                        sweep_lidar_info['timestamp'] = sweep_lidar_data[
                            'timestamp']
                        sweep_lidar_info['is_key_frame'] = sweep_lidar_data[
                            'is_key_frame']
                        sweep_lidar_info['filename'] = sweep_lidar_data[
                            'filename']
                        sweep_lidar_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info
            # Remove empty sweeps.
            for i, sweep in enumerate(cam_sweeps):
                if len(sweep.keys()) == 0:
                    cam_sweeps = cam_sweeps[:i]
                    break
            for i, sweep in enumerate(lidar_sweeps):
                if len(sweep.keys()) == 0:
                    lidar_sweeps = lidar_sweeps[:i]
                    break
            info['cam_sweeps'] = cam_sweeps
            info['lidar_sweeps'] = lidar_sweeps
            ann_infos = list()

            if 'anns' in cur_sample:
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def main():
    nusc = NuScenes(version='v1.0-trainval',
                             dataroot='data/nuScenes',
                             verbose=True)
    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]

    # 过滤出用户定义的天气场景
    sun_scenes = val_sun
    night_scenes = val_night
    rain_scenes = val_rain
    # train_scenes = splits.train
    # val_scenes = splits.val
    # train_infos_tiny = generate_info(trainval_nusc, train_scenes[:2])
    # mmcv.dump(train_infos_tiny, 'data/nuScenes/nuscenes_infos_train-tiny.pkl')
    sun_infos = generate_info(nusc, sun_scenes)
    mmcv.dump(sun_infos, 'data/nuScenes/nuscenes_weather_sun_val.pkl')
    night_infos = generate_info(nusc, night_scenes)
    mmcv.dump(night_infos, 'data/nuScenes/nuscenes_weather_night_val.pkl')
    rain_infos = generate_info(nusc, rain_scenes)
    mmcv.dump(rain_infos, 'data/nuScenes/nuscenes_weather_rain_val.pkl')

    # val_infos = generate_info(trainval_nusc, val_scenes)
    # mmcv.dump(val_infos, 'data/nuScenes/nuscenes_infos_val.pkl')

    # test_nusc = NuScenes(version='v1.0-test',
    #                      dataroot='./data/nuScenes/v1.0-test/',
    #                      verbose=True)
    # test_scenes = splits.test
    # test_infos = generate_info(test_nusc, test_scenes)
    # mmcv.dump(test_infos, './data/nuScenes/nuscenes_infos_test.pkl')


if __name__ == '__main__':
    main()
