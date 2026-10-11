import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthGuidedAttention(nn.Module):
    """深度引导的注意力模块，利用radar深度信息构建注意力图"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 深度特征转换
        self.depth_conv = nn.Sequential(
            nn.Conv2d(70, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1)
        )

        # 处理特征到目标高度的上采样层
        self.height_upsample = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=(16, 1), stride=(16, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # 自适应权重生成
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels + 16, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_info):  #24,80,70,44
        B = rgb_feat.size(0)

        # 处理深度信息
        depth_feat = depth_info.mean(dim=1)  # [B*N, D, W]
        depth_feat = depth_feat.unsqueeze(1).permute(0,2,1,3)  # [B*N, 1, D, W]
        depth_feat = self.depth_conv(depth_feat) # [B*N, 16, 1, W]
        depth_feat = self.height_upsample(depth_feat)
        # 生成注意力权重
        concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)
        attention = self.weight_net(concat_feat)

        return rgb_feat * attention


class MultiScaleFeatureFusion(nn.Module):
    """多尺度高度特征融合模块"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 多尺度特征提取
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=r, dilation=r)
            for r in [1, 2, 4, 8]
        ])

        # 特征聚合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1)
        )

    def forward(self, x):
        multi_scale_feats = [conv(x) for conv in self.scale_convs]
        concat_feat = torch.cat(multi_scale_feats, dim=1)
        return self.fusion_conv(concat_feat)



class ImprovedFeatureFusion(nn.Module):
    def __init__(self, rgb_channels=512,radar_channels=80, num_heads=8):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.radar_channels = radar_channels

        # 1. 特征预处理
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 1),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU()
        )

        self.radar_conv = nn.Sequential(
            nn.Conv2d(radar_channels, radar_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(radar_channels),
            nn.ReLU()
        )

        # 2. 深度引导注意力
        self.depth_attention = DepthGuidedAttention(rgb_channels)

        # 3. 多尺度特征融合
        self.multi_scale = MultiScaleFeatureFusion(rgb_channels)

        # 4. 3D位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, rgb_channels, 16, 44))

        # 5. 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels, rgb_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(rgb_channels // 4, rgb_channels, 1),
            nn.Sigmoid()
        )

        # 6. 输出处理
        self.output_conv = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 1),
            nn.BatchNorm2d(rgb_channels),
            nn.Sigmoid()
        )


    def forward(self, rgb_feat, radar_feat):
        """
        Args:
            rgb_feat: [B*N, C, H, W]
            radar_feat: [B*N, C, D, W]
        """
        # 1. 特征预处理
        rgb_feat = self.rgb_conv(rgb_feat)
        radar_feat = self.radar_conv(radar_feat)

        # 2. 添加3D位置编码
        rgb_feat = rgb_feat + self.pos_embedding

        # 3. 深度引导的注意力增强
        enhanced_rgb = self.depth_attention(rgb_feat, radar_feat)

        # 4. 多尺度特征融合
        multi_scale_feat = self.multi_scale(enhanced_rgb)

        # 5. 通道注意力
        channel_weights = self.channel_attention(multi_scale_feat)
        weighted_feat = multi_scale_feat * channel_weights

        # 6. 特征融合
        concat_feat = torch.cat([weighted_feat, rgb_feat], dim=1)
        output = self.output_conv(concat_feat)


        return output


