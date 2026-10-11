import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from .gaussian import *
from mmdet3d.ops import DynamicScatter,ball_query,gather_points
from mmdet3d.ops import grouping_operation
from mmdet3d.models.utils import MLP
from mmcv.runner import force_fp32
# Initialize a Registry
from mmcv.utils import Registry

class RadarDenseEncoder(nn.Module):
    def __init__(self, in_channels=40):
        super(RadarDenseEncoder, self).__init__()
        self.rcs_att = nn.Conv2d(2, in_channels, 1)
        self.rcs_gamma = nn.Parameter(torch.tensor(1.0))


    def forward(self, voxels, batch_size, B,  rcoors):
        # 初始化 RCS 相关特征
        heatmap = voxels.new_zeros((batch_size, 128, 128))
        heatmap_feat = voxels.new_zeros((batch_size, 1, 128, 128))

        rcs =  voxels[:, :, 3].mean(dim=1)
        true_rcs = torch.nn.functional.relu(rcs)
        radius = self.rcs_gamma * true_rcs + 1

        for i in range(rcoors.shape[0]):
            batch, _, y, x = rcoors[i]
            draw_heatmap_gaussian_rcs(heatmap[batch], [x, y], int(radius[i].data.item()))
            heatmap_feat[batch] = draw_heatmap_gaussian_feat(heatmap_feat[batch], [x, y], int(radius[i].data.item()),
                                                             true_rcs[i])
        rcs_features = self.rcs_att(torch.cat([heatmap.unsqueeze(dim=1), heatmap_feat], dim=1))
        rcs_features = rcs_features.view(B, 6, 40, 128, 128).permute(0, 2, 1, 3, 4).reshape(B, 240, 128, 128)

        return rcs_features  # 返回 (4, 240, 128, 128)
