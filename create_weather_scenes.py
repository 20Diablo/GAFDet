import mmcv
import numpy as np
import os
import re
from os import path as osp
from pyquaternion import Quaternion
from mmdet3d.datasets import NuScenesDataset
from radar_converter_coordinate_v1 import get_available_scenes,obtain_sensor2top
from nuscenes.utils import splits
from nuscenes.nuscenes import NuScenes
"""
# sun in val 123
val_sun = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565','scene-0770', 'scene-0771',
     'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780','scene-0781', 'scene-0782', 'scene-0783', 'scene-0784',
     'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797','scene-0798', 'scene-0799', 'scene-0800', 'scene-0802',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972','scene-1059',
     'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1066','scene-1068', 'scene-1069','scene-1070',
     'scene-1071', 'scene-1072', 'scene-1073']
# heavy_rain in val 27
val_rain = \
    ['scene-0625', 'scene-0626','scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634',
     'scene-0635', 'scene-0636','scene-0637', 'scene-0638','scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-1060', 'scene-1065', 'scene-1067']
"""
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


def get_scenes_categories():
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuScenes', verbose=True)
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    scene_categories = {}
    for scene in nusc.scene:
        scene_name = scene['name']
        description = scene['description'].lower()  # 转换为小写以进行匹配
        # 使用正则表达式提取类别信息
        """
        ### heavy 好像是指车流量大
        if description.startswith("heavy,"):  # scene-0646描述是以 "heavy," 开头，应归为 "rain" 类别
            category = 'rain'
        elif re.search(r'\brain\b', description):
            category = 'rain'
        else:
            category = 'sun'
        """
        if re.search(r'\brain\b', description):
            category = 'rain'
        else:
            category = 'sun'
        scene_categories[scene_name] = category
    for scene_name, category in scene_categories.items():
        print(f"Scene {scene_name}: {category}")

def _fill_trainval_infos(nusc,
                         sun_scenes,
                         night_scenes,
                         rain_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.
    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.
    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    sun_infos, night_infos, rain_infos = [], [], []

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        scene_token = sample['scene_token']
        if scene_token in sun_scenes:
            sun_infos.append(info)
        elif scene_token in night_scenes:
            night_infos.append(info)
        elif scene_token in rain_scenes:
            rain_infos.append(info)

    return sun_infos, night_infos, rain_infos


def create_nuscenes_weather_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]

    # 过滤出用户定义的天气场景
    sun_scenes = [s['token'] for s in available_scenes if s['name'] in val_sun]
    night_scenes = [s['token'] for s in available_scenes if s['name'] in val_night]
    rain_scenes = [s['token'] for s in available_scenes if s['name'] in val_rain]

    # 生成三个天气场景的信息列表
    sun_infos, night_infos, rain_infos = _fill_trainval_infos(
        nusc, sun_scenes, night_scenes, rain_scenes, max_sweeps=max_sweeps
    )

    # 保存为.pkl文件
    metadata = dict(version=version)
    mmcv.dump({'infos': sun_infos, 'metadata': metadata}, osp.join(root_path, f'{info_prefix}_sun_val.pkl'))
    mmcv.dump({'infos': night_infos, 'metadata': metadata}, osp.join(root_path, f'{info_prefix}_night_val.pkl'))
    mmcv.dump({'infos': rain_infos, 'metadata': metadata}, osp.join(root_path, f'{info_prefix}_rain_val.pkl'))

    print(f'Sun Val Scenes: {len(sun_infos)} samples')
    print(f'Night Val Scenes: {len(night_infos)} samples')
    print(f'Rain Val Scenes: {len(rain_infos)} samples')

if __name__ == '__main__':
    #get_scenes_categories()
    #weather = 'rain' 'night' 'rain_night'
    create_nuscenes_weather_infos('data/nuScenes/', 'nuscenes_weather', version='v1.0-trainval')
