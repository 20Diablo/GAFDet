import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import mmcv
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# 导入你的模型和数据集
from datasets.nusc_det_dataset import NuscDatasetRadarDet
from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel


class CRNFeatureVisualizer:
    def __init__(self, model_path, device='cuda:0'):
        """
        初始化CRN特征可视化器

        Args:
            model_path: 训练好的模型权重路径(.ckpt文件)
            device: 推理设备
        """
        self.device = device

        # 加载模型
        self.lightning_model = self._load_model(model_path)
        self.model = self.lightning_model.model
        self.model.eval()

        # 特征钩子存储
        self.features = {}
        self._register_hooks()

    def _load_model(self, model_path):
        """加载训练好的模型"""
        lightning_model = CRNLightningModel.load_from_checkpoint(
            model_path,
            map_location=self.device
        )
        lightning_model.to(self.device)
        lightning_model.eval()
        return lightning_model

    def _register_hooks(self):
        """注册前向钩子以提取MFAFuser中的关键特征"""
        # 在MFAFuser的_forward_single_sweep方法上注册钩子
        fuser = self.model.fuser

        # 注册钩子到_forward_single_sweep方法
        original_forward_single_sweep = fuser._forward_single_sweep

        def hooked_forward_single_sweep(feat_img, feat_pts):
            # 保存输入特征
            self.features['feat_img'] = feat_img.detach().cpu()
            self.features['feat_pts'] = feat_pts.detach().cpu()

            # 调用原始方法
            output = original_forward_single_sweep(feat_img, feat_pts)

            # 保存输出特征
            self.features['fused_output'] = output.detach().cpu()

            return output

        # 替换方法
        fuser._forward_single_sweep = hooked_forward_single_sweep

    def extract_features(self, batch_data):
        """提取所有中间特征"""
        self.features.clear()

        with torch.no_grad():
            # 准备数据
            sweep_imgs, mats, img_metas, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv = batch_data

            if torch.cuda.is_available():
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
                pts_pv = pts_pv.cuda()

            # 前向传播，触发钩子
            _ = self.lightning_model(sweep_imgs, mats, pts_pv=pts_pv, is_train=False)

        return self.features

    def visualize_bev_heatmap(self, feature_map, ax, title, channel_idx=0, cmap='viridis'):
        """可视化BEV特征热力图"""
        if len(feature_map.shape) == 4:
            # 取第一个batch和指定通道
            heatmap = feature_map[0, channel_idx].numpy()
        else:
            heatmap = feature_map.numpy()

        im = ax.imshow(heatmap, cmap=cmap, origin='lower')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # 添加colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        return im

    def visualize_sample(self, sample_idx, dataset, output_dir):
        """可视化单个样本的所有特征"""
        print(f"Visualizing sample {sample_idx}...")

        # 获取数据 - 数据集返回的是列表
        batch_data = dataset[sample_idx]

        # 根据数据集的__getitem__方法，解析返回的列表
        sweep_imgs = batch_data[0]  # sweep_imgs
        sweep_sensor2ego_mats = batch_data[1]  # sweep_sensor2ego_mats
        sweep_intrins = batch_data[2]  # sweep_intrins
        sweep_ida_mats = batch_data[3]  # sweep_ida_mats
        sweep_sensor2sensor_mats = batch_data[4]  # sweep_sensor2sensor_mats
        bda_mat = batch_data[5]  # bda_mat
        sweep_timestamps = batch_data[6]  # sweep_timestamps
        img_metas = batch_data[7]  # img_metas
        gt_boxes_3d = batch_data[8]  # gt_boxes_3d
        gt_labels_3d = batch_data[9]  # gt_labels_3d
        depth_labels = batch_data[10]  # depth_labels (可能为None)
        pts_pv = batch_data[11]  # radar_pv

        sweep_imgs = sweep_imgs.unsqueeze(0).float()  # 添加 .float()
        if len(pts_pv.shape) == 3:  # [num_cams, num_points, features]
            pts_pv = pts_pv.unsqueeze(0).unsqueeze(0).float()  # 添加 batch 和 sweep 维度
        elif len(pts_pv.shape) == 4:  # [num_sweeps, num_cams, num_points, features]
            pts_pv = pts_pv.unsqueeze(0).float()  # 只添加 batch 维度

        # 构建mats_dict
        mats_dict = {
            'sensor2ego_mats': sweep_sensor2ego_mats.unsqueeze(0).float(),
            'intrin_mats': sweep_intrins.unsqueeze(0).float(),
            'ida_mats': sweep_ida_mats.unsqueeze(0).float(),
            'sensor2sensor_mats': sweep_sensor2sensor_mats.unsqueeze(0).float(),
            'bda_mat': bda_mat.unsqueeze(0).float()
        }

        # 添加batch维度
        sweep_imgs = sweep_imgs.unsqueeze(0)
        pts_pv = pts_pv.unsqueeze(0)

        # 提取特征
        batch_data_formatted = (sweep_imgs, mats_dict, img_metas, gt_boxes_3d, gt_labels_3d, None, depth_labels, pts_pv)
        features = self.extract_features(batch_data_formatted)

        # 创建输出目录
        sample_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_dir, exist_ok=True)

        # 创建可视化图
        fig = plt.figure(figsize=(18, 12))

        # 创建新的子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 第一行：前3个相机
        IMG_KEYS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            img = sweep_imgs[0, 0, i].cpu().numpy().transpose(1, 2, 0)
            # 使用正确的归一化参数
            mean = np.array([123.675, 116.28, 103.53]) / 255.0
            std = np.array([58.395, 57.12, 57.375]) / 255.0
            img = img * std + mean
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(IMG_KEYS[i])
            ax.axis('off')

        # 第二行：后3个相机
        IMG_KEYS = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            img = sweep_imgs[0, 0, i + 3].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([123.675, 116.28, 103.53]) / 255.0
            std = np.array([58.395, 57.12, 57.375]) / 255.0
            img = img * std + mean
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(IMG_KEYS[i])
            ax.axis('off')

        # 第三行：特征热力图
        feature_titles = ['Image BEV Features', 'Radar BEV Features', 'Fused BEV Features']
        feature_keys = ['feat_img', 'feat_pts', 'fused_output']
        cmaps = ['viridis', 'plasma', 'inferno']

        for i, (key, title, cmap) in enumerate(zip(feature_keys, feature_titles, cmaps)):
            if key in features:
                ax = fig.add_subplot(gs[2, i])
                self.visualize_bev_heatmap(features[key], ax, title, channel_idx=0, cmap=cmap)
                print(f"{key} shape: {features[key].shape}")
            else:
                print(f"Warning: {key} not found in features")

        plt.suptitle(f'Sample {sample_idx} - CRN Feature Visualization', fontsize=16)

        # 保存图像
        save_path = os.path.join(sample_dir, 'crn_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 单独保存每个特征的详细热力图
        for key, title, cmap in zip(feature_keys, feature_titles, cmaps):
            if key in features:
                self.save_detailed_heatmap(features[key], title,
                                           os.path.join(sample_dir, f'{key}_detailed.png'),
                                           cmap=cmap)

        print(f"Features saved to: {sample_dir}")

    def save_detailed_heatmap(self, feature_map, title, save_path, num_channels=8, cmap='viridis'):
        """保存详细的多通道特征热力图"""
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]  # 取第一个batch

        C, H, W = feature_map.shape
        num_channels = min(num_channels, C)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(num_channels):
            heatmap = feature_map[i].numpy()
            im = axes[i].imshow(heatmap, cmap=cmap, origin='lower')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)

        # 隐藏多余的子图
        for i in range(num_channels, 8):
            axes[i].axis('off')

        plt.suptitle(f'{title} - Multi-Channel View', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_dataset_with_correct_config(data_root):
    """使用正确的配置创建数据集"""

    # 直接使用base_exp.py中的配置
    img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                    img_std=[58.395, 57.12, 57.375],
                    to_rgb=True)

    ida_aug_conf = {
        'resize_lim': (0.386, 0.55),
        'final_dim': (256, 704),
        'rot_lim': (0., 0.),  # 关闭旋转用于可视化
        'H': 900,
        'W': 1600,
        'rand_flip': False,  # 关闭随机翻转用于可视化
        'bot_pct_lim': (0.0, 0.0),
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6,
    }

    bda_aug_conf = {
        'rot_ratio': 0.0,  # 关闭旋转用于可视化
        'rot_lim': (0.0, 0.0),
        'scale_lim': (1.0, 1.0),
        'flip_dx_ratio': 0.0,
        'flip_dy_ratio': 0.0
    }

    rda_aug_conf = {
        'N_sweeps': 6,
        'N_use': 5,
        'drop_ratio': 0.0,  # 关闭drop用于可视化
    }

    backbone_img_conf = {
        'x_bound': [-51.2, 51.2, 0.8],
        'y_bound': [-51.2, 51.2, 0.8],
        'z_bound': [-5, 3, 8],
        'd_bound': [2.0, 58.0, 0.8],
        'final_dim': (256, 704),
        'output_channels': 80,
        'downsample_factor': 16,
    }

    classes = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    ]

    dataset = NuscDatasetRadarDet(
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        rda_aug_conf=rda_aug_conf,
        img_backbone_conf=backbone_img_conf,
        classes=classes,
        data_root=data_root,
        info_paths='data/nuScenes/nuscenes_infos_val.pkl',
        is_train=False,
        img_conf=img_conf,
        load_interval=1,
        num_sweeps=4,
        sweep_idxes=[0, 1, 2, 3],
        key_idxes=[-2, -4, -6],
        return_image=True,
        return_depth=False,
        return_radar_pv=True,
        remove_z_axis=False,
        radar_pv_path='radar_pv_filter',
    )

    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True,
                        help='Path to trained model checkpoint (.ckpt file)')
    parser.add_argument('--data_root', required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--output_dir', default='./crn_feature_visualizations',
                        help='Output directory')
    parser.add_argument('--sample_indices', nargs='+', type=int, default=[0, 100, 200],
                        help='Sample indices to visualize')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')

    args = parser.parse_args()

    # 创建可视化器
    print("Loading model...")
    visualizer = CRNFeatureVisualizer(args.model_path, args.device)

    # 加载数据集 - 使用正确的配置
    print("Loading dataset...")
    dataset = create_dataset_with_correct_config(args.data_root)

    print(f"Dataset loaded: {len(dataset)} samples")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 可视化指定样本
    for sample_idx in args.sample_indices:
        if sample_idx < len(dataset):
            visualizer.visualize_sample(sample_idx, dataset, args.output_dir)
        else:
            print(f"Warning: Sample index {sample_idx} exceeds dataset size {len(dataset)}")

    print(f"All visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()