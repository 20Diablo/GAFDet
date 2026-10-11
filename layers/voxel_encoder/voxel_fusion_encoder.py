# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from mmcv.runner import force_fp32
from torch import nn
import numpy as np
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmdet3d.models.voxel_encoders.utils import PFNLayer, get_paddings_indicator

@VOXEL_ENCODERS.register_module()
class PillarFusionFeatureNet(nn.Module):

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(PillarFusionFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        self.in_channels = in_channels
        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        self.lidar_proj_layer = nn.Linear(2, 2, bias=False)  # training log 18, no bias
        nn.init.normal_(self.lidar_proj_layer.weight, mean=0, std=0.01)
        self.compress = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.rcs_att = nn.Conv2d(2, in_channels, 1)
        self.rcs_gamma = nn.Parameter(torch.tensor(1.0))
        self.voxel_size = [0.1, 0.1, 0.2]
        self.radar_voxel_size = [0.2, 0.2, 0.2]
        self.point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        mask_radar = features[:, :, 9] == 1
        mask_voxels_radar = mask_radar.any(dim=1)
        coors = coors[mask_voxels_radar]
        features = features[mask_voxels_radar]
        num_points = num_points[mask_voxels_radar]
        dtype = features.dtype

        ## 处理lidar、radar独特的属性
        # for lidar: x,y,z,internsity,dt,0,0,0,0,-1
        # for radar: x,y,z,0,0,rcs,vx_comp,vy_comp,dt,1
        # return: x,y,z,internsity,dt,rcs,vx_comp,vy_comp,dt,point -> pillar
        out = torch.zeros((features.shape[0], self.in_channels), dtype=torch.float32, device=features.device)
        out[:, :2] = features[:, :, :2].sum(dim=1) / num_points.view(-1, 1).float()  # average the xy

        mask_radar = features[:, :, 9] == 1
        mask_radar_sum = mask_radar.sum(dim=1)
        mask_radar_sum_modified = torch.clamp(mask_radar_sum, min=1)
        out[:, 5:9] = torch.sum(features[:, :, 5:9] * mask_radar.unsqueeze(2), dim=1) / mask_radar_sum_modified.view(-1,
                                                                                                                   1).float()
        mask_lidar = features[:, :, 9] == -1
        if torch.sum(mask_lidar) > 0:
            mask_lidar_sum = mask_lidar.sum(dim=1)
            mask_lidar_sum_modified = torch.clamp(mask_lidar_sum, min=1)
            ### 通过lidar点的高斯加权平均 赋值pillar的高度
            heights = features[:, :, 2] * mask_lidar
            for i in range(heights.shape[0]):
                height = heights[i]
                height = height[ height != 0]
                if len(height) > 1 :
                    mean = height.mean()
                    std = height.std()
                    if std == 0:
                        out[i, 2] = height[0]
                    else:
                        weights = torch.exp(-0.5 * ((height - mean) / std) ** 2)
                        weights /= weights.sum()  # 归一化权重
                        out[i, 2] = (height * weights).sum()
                elif len(height) == 1:
                    out[i, 2] = height[0]
                else:
                    out[i, 2] = torch.tensor(0, dtype=out.dtype)
            out[:, 3:5] = self.lidar_proj_layer(
                torch.sum(features[:, :, 3:5] * mask_lidar.unsqueeze(2), dim=1) / mask_lidar_sum_modified.view(-1,
                                                                                                             1).float())  # average the intensity and dt
            ### 9:12 lidar -> point mean 12:15 lidar -> pillar center
            lidar_mean = torch.sum(features[:, :, :3] * mask_lidar.unsqueeze(2), dim=1) / mask_lidar_sum_modified.view(
                -1, 1).float()
            ### 过滤没有lidar的pillar
            mask_voxels_lidar = mask_lidar.any(dim=1)
            out[mask_voxels_lidar, 9:12] = lidar_mean[mask_voxels_lidar, :3] - out[mask_voxels_lidar, :3]
            out[mask_voxels_lidar, 12] = lidar_mean[mask_voxels_lidar, 0] - (
                        coors[mask_voxels_lidar, 3].to(dtype) * self.vx + self.x_offset)
            out[mask_voxels_lidar, 13] = lidar_mean[mask_voxels_lidar, 1] - (
                        coors[mask_voxels_lidar, 2].to(dtype) * self.vy + self.y_offset)
            out[mask_voxels_lidar, 14] = lidar_mean[mask_voxels_lidar, 2] - (
                        coors[mask_voxels_lidar, 1].to(dtype) * self.vz + self.z_offset)
        else:
            # print("no radar points") # if the first frame of a scene, no lidar points
            # avoid error when training with b_size 1 on multi GPU
            out[:, 2] = torch.zeros((out.shape[0]), device=features.device)
            out[:, 3:5] = torch.zeros((out.shape[0], 2), device=features.device)
            out[:, 9:15] = torch.zeros((out.shape[0], 6), device=features.device)
        out[:, 15] = torch.zeros((out.shape[0]), device=features.device)
        return out, num_points, coors

@VOXEL_ENCODERS.register_module()
class PillarFusionFeatureNet_point(nn.Module):

    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(PillarFusionFeatureNet_point, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        mask_radar = features[:, :, 9] == 1
        mask_voxels_radar = mask_radar.any(dim=1)
        coors = coors[mask_voxels_radar]
        features = features[mask_voxels_radar]
        num_points = num_points[mask_voxels_radar]

        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        mask_radar = features[:, :, 9] == 1
        mask_radar_sum = mask_radar.sum(dim=1)
        mask_radar_sum_modified = torch.clamp(mask_radar_sum, min=1)
        rcs = torch.sum(features[:, :, 5].unsqueeze(2) * mask_radar.unsqueeze(2), dim=1) / mask_radar_sum_modified.view(-1,1).float()

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return rcs, features.squeeze(1), num_points, coors
