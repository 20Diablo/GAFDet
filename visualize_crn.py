import os
import math
from argparse import ArgumentParser

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from nuscenes.utils.data_classes import Box, LidarPointCloud
from pyquaternion import Quaternion

from datasets.nusc_det_dataset import map_name_from_general_to_detection


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('idx',
                        type=int,
                        help='Index of the dataset to be visualized.')
    parser.add_argument('result_path', help='Path of the result json file.')
    parser.add_argument('target_path', help='Target path to save the visualization result.')

    args = parser.parse_args()
    return args


def get_ego_box(box_dict, ego2global_rotation, ego2global_translation):
    box = Box(
        box_dict['translation'],
        box_dict['size'],
        Quaternion(box_dict['rotation']),
    )
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    box.translate(trans)
    box.rotate(rot)
    box_xyz = np.array(box.center)
    box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
    box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
    box_velo = np.array(box.velocity[:2])
    return np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center
    Returns:
    """
    template = (np.array((
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    )) / 2)

    corners3d = np.tile(boxes3d[:, None, 3:6], (1, 8, 1)) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def get_bev_lines(corners):
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]],
             [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                   [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]], [corners[st, 1], corners[ed, 1]]])
    return ret


def get_cam_corners(corners, translation, rotation, cam_intrinsics):
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners


def demo(
        info,
        idx,
        nusc_results_file,
        dump_file,
        threshold=0.22,
        show_range=60,
        show_classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ],
):
    # Set cameras
    IMG_KEYS = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    # Create dump_file directory if it doesn't exist
    if not os.path.exists(dump_file):
        os.makedirs(dump_file)

    # Get data from dataset
    results = mmcv.load(nusc_results_file)['results']
    lidar_path = info['lidar_infos']['LIDAR_TOP']['filename']
    # 修改为加载radar BEV数据而不是LiDAR数据
    radar_bev_path = os.path.join('data/nuScenes/radar_bev_filter', os.path.split(lidar_path)[-1])

    # 加载radar BEV点云数据 (格式: x, y, z, rcs, vx_comp, vy_comp, sweep_info)
    radar_points = np.fromfile(radar_bev_path, dtype=np.float32, count=-1).reshape(-1, 7)

    # Get point cloud - 使用radar数据而不需要LiDAR的标定传感器变换
    pts = radar_points.copy()
    ego2global_rotation = np.mean(
        [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in IMG_KEYS], 0)
    ego2global_translation = np.mean([
        info['cam_infos'][cam]['ego_pose']['translation'] for cam in IMG_KEYS], 0)
    # lidar_points = LidarPointCloud(lidar_points.T)
    # lidar_points.rotate(
    #     Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    # lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))
    # pts = lidar_points.points.T

    # Get GT corners
    gt_corners = []
    for i in range(len(info['ann_infos'])):
        if map_name_from_general_to_detection[
            info['ann_infos'][i]['category_name']] in show_classes:
            box = get_ego_box(
                dict(
                    size=info['ann_infos'][i]['size'],
                    rotation=info['ann_infos'][i]['rotation'],
                    translation=info['ann_infos'][i]['translation'],
                ), ego2global_rotation, ego2global_translation)
            if np.linalg.norm(box[:2]) <= show_range:
                corners = get_corners(box[None])[0]
                gt_corners.append(corners)

    # Get prediction corners
    pred_corners, pred_class, pred_distance, pred_degrees = [], [], [], []
    for box in results[info['sample_token']]:
        if box['detection_score'] >= threshold and box['detection_name'] in show_classes:
            box3d = get_ego_box(box, ego2global_rotation, ego2global_translation)
            box3d[2] += 0.5 * box3d[5]  # NOTE
            if np.linalg.norm(box3d[:2]) <= show_range:
                corners = get_corners(box3d[None])[0]
                pred_corners.append(corners)
                pred_class.append(box['detection_name'])
                # pred_distance.append(np.linalg.norm(box3d[:2]))  # 단위: m
                # theta_radians = math.atan2(box3d[1], box3d[0])  # 라디안 단위 각도
                # theta_degrees = math.degrees(theta_radians)  # 각도로 변환
                # # 0~359°로 정규화
                pred_distance.append(np.linalg.norm(box3d[:2]))  # 单位: m
                theta_radians = math.atan2(box3d[1], box3d[0])  # 弧度单位角度
                theta_degrees = math.degrees(theta_radians)  # 角度转换
                # 0~359°归一化
                theta_normalized = (theta_degrees + 360) % 360
                pred_degrees.append(theta_normalized)

    # Set figure size
    plt.figure(figsize=(24, 8))
    # plt.figure(figsize=(16, 16))

    for i, k in enumerate(IMG_KEYS):
        # Draw camera views
        fig_idx = i + 1 if i < 3 else i + 3
        plt.subplot(2, 5, fig_idx)

        # Set camera attributes
        # plt.title(k)
        plt.axis('off')
        plt.xlim(0, 1600)
        plt.ylim(900, 0)

        img = mmcv.imread(
            os.path.join('data/nuScenes', info['cam_infos'][k]['filename']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw images
        plt.imshow(img)

        # # Draw 3D predictions
        # for corners, cls in zip(pred_corners, pred_class):
        #     cam_corners = get_cam_corners(
        #         corners,
        #         info['cam_infos'][k]['calibrated_sensor']['translation'],
        #         info['cam_infos'][k]['calibrated_sensor']['rotation'],
        #         info['cam_infos'][k]['calibrated_sensor']['camera_intrinsic'])
        #     lines = get_3d_lines(cam_corners)
        #     for line in lines:
        #         plt.plot(line[0],
        #                  line[1],
        #                  c=cm.get_cmap('tab10')(show_classes.index(cls)))

        # --- 新增：绘制原始Radar BEV点云图 (不带真值和预测) ---
    plt.subplot(1, 5, 4)  # 放在第2行第4列的位置（可根据需要调整）
    plt.title('Raw Radar BEV')
    plt.axis('equal')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('black')
    plt.xlim(-45, 45)
    plt.ylim(-45, 45)
    # 绘制原始radar点云，使用单一颜色
    plt.scatter(-pts[:, 1], pts[:, 0], s=0.5, c='gray', alpha=0.8)

    # Draw BEV
    plt.subplot(1,5,5)
    # plt.subplot(1, 1, 1)

    # Set BEV attributes
    plt.title('BEV: RADAR')
    plt.axis('equal')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('black')
    plt.xlim(-45, 45)
    plt.ylim(-45, 45)
    # plt.xlim(-40, 40)
    # plt.ylim(-40, 40)

    # plt.tick_params(left=False, right=False, labelleft=False, labelright=False)
    # for spine in plt.gca().spines.values():
    #     spine.set_visible(True)
    #     spine.set_linewidth(1)
    #     spine.set_color('black')
    # Draw point cloud
    # plt.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap='gray')
    # Draw radar point cloud - 使用单一颜色
    plt.scatter(-pts[:, 1], pts[:, 0], s=0.5, c='gray', alpha=0.8)

    # Draw BEV GT boxes
    for corners in gt_corners:
        lines = get_bev_lines(corners)
        for line in lines:
            plt.plot([-x for x in line[1]],
                     line[0],
                     c='g',
                     label='ground truth')

    myfont = {'family': 'serif',
              # 'color': 'forestgreen',
              'color': 'black',
              'weight': 'normal',
              'size': 8}
    margin = 0.2

    # Draw BEV predictions
    for corners, cls, dist, degree in zip(pred_corners, pred_class, pred_distance, pred_degrees):
        lines = get_bev_lines(corners)
        for line in lines:
            plt.plot([-x for x in line[1]], line[0], c='b', label='prediction')
    #     # # todo: show direction and distance(from center)
    #     # plt.text(-line[1][1] + margin, line[0][1] + margin,
    #     #          str(round(dist)) + 'm,' + str(round(360-degree)) + '°',
    #     #          fontdict=myfont)

    # Set legend
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(),
    #            by_label.keys(),
    #            loc='upper right',
    #            framealpha=1,
    #            fontsize=10)

    # Save figure
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    # plt.subplots_adjust(wspace=0.3, hspace=0.1)
    plt.savefig(f'{dump_file}index-{idx}.png')
    plt.close()


if __name__ == '__main__':
    # args = parse_args()

    infos = mmcv.load('data/nuScenes/nuscenes_infos_val.pkl')
    # idx = len(infos)
    # print(idx)
    # Centers: [100, 300, 1000, 2500, 3000, 4000, 5700, 5800]
    # Each with ~5 frames, excluding [2406, 2670, 2832, 4104, 5634, 5700]
    target_ids = [
        91, 103, 108, 112, 115, 
        285, 298, 304, 311, 336, 
        985, 992, 1006, 1011, 1018, 
        2488, 2492, 2503, 2510, 2515, 
        2982, 2995, 3001, 3008, 3019, 
        3985, 3996, 4003, 4018, 4022,
        5682, 5691, 5704, 5712, 5718, 
        5785, 5792, 5803, 5811, 5816
    ]
    for i in target_ids:
        # if i > 1:
        #     break

        demo(
            # args.idx,
            # args.result_path,
            # args.target_path,
            infos[i],
            i,
            './outputs/det/CRN_r50_256x704_128x128_4key/results_nusc.json',
            './vis_RC_for_dj/',
        )
        