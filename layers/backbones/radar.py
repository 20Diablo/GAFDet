import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from mmcv.cnn import build_norm_layer


class RadarHistoryFusion(nn.Module):
    def __init__(self, feature_dim=80, history_frames=3, use_gate=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.history_frames = history_frames
        self.history_queue = []
        self.use_gate = use_gate

        # 光流估计网络
        self.flow_estimation = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)  # 输出光流场 (vx, vy)
        )

        # 可变形对齐模块
        self.deform_align = torchvision.ops.DeformConv2d(
            feature_dim, feature_dim, kernel_size=3, padding=1, groups=4
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, batch_first=True
        )

        # 通道注意力模块
        self.channel_att = ChannelAttention3D(embed_dims=feature_dim * (history_frames + 1))

        # 特征压缩
        self.reduce_conv = nn.Sequential(
            nn.Conv3d(feature_dim * (history_frames + 1), feature_dim * 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(feature_dim * 2),
            nn.ReLU(),
            nn.Conv3d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU()
        )

        # 门控机制
        if self.use_gate:
            self.gate_net = nn.Sequential(
                nn.Conv2d(feature_dim * 2, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()
            )

    def align_features(self, hist_feat, flow):
        """使用可变形卷积对齐历史帧"""
        B, T, C, H, W = hist_feat.shape

        # 分解偏移量
        offsets = flow.view(B, T, 2 * 3 * 3 * 4, H, W)

        # 逐帧对齐
        aligned_feats = []
        for t in range(T):
            offset_t = offsets[:, t].view(B, -1, H, W)
            aligned = self.deform_align(
                hist_feat[:, t].contiguous(),
                offset_t
            )
            aligned_feats.append(aligned.unsqueeze(1))

        return torch.cat(aligned_feats, dim=1)  # [B, T, C, H, W]

    def compute_attention_weights(self, curr_feat, hist_feat):
        """使用注意力机制计算历史帧权重"""
        B, T, C, H, W = curr_feat.shape

        # 将特征展平为序列
        curr_seq = curr_feat.view(B, T, C, -1).permute(0, 1, 3, 2)  # [B, T, H*W, C]
        hist_seq = hist_feat.view(B, T, C, -1).permute(0, 1, 3, 2)  # [B, T, H*W, C]

        # 计算注意力权重
        attn_output, attn_weights = self.attention(
            query=curr_seq, key=hist_seq, value=hist_seq
        )

        # 恢复形状
        attn_weights = attn_weights.view(B, T, H, W)  # [B, T, H, W]
        return attn_weights.unsqueeze(2)  # [B, T, 1, H, W]

    def forward(self, x):
        """
        输入: x shape is (B, num_sweep, C, H, W)
        输出: (B, num_sweep, C, H, W)
        """
        B, T, C, H, W = x.shape

        # 对前几帧，直接返回原始特征
        if len(self.history_queue) < self.history_frames:
            self.history_queue.append(x.clone().detach())
            return x

        # 获取当前帧
        curr_frame = x

        # 对每个历史帧进行运动对齐和加权
        weighted_history_frames = []
        for hist_frame in self.history_queue:
            # 光流估计
            flow = self.flow_estimation(
                torch.cat([curr_frame.mean(dim=1), hist_frame.mean(dim=1)], dim=1)
            )

            # 可变形对齐
            aligned_hist = self.align_features(hist_frame, flow)

            # 使用注意力机制计算权重
            weights = self.compute_attention_weights(curr_frame, aligned_hist)
            weighted_hist = aligned_hist * weights
            weighted_history_frames.append(weighted_hist)

        # 拼接所有特征
        fused_features = torch.cat([curr_frame] + weighted_history_frames, dim=2)

        # 3D通道注意力处理
        enhanced_features = fused_features.permute(0, 2, 1, 3, 4)
        enhanced_features = self.channel_att(enhanced_features)

        # 通道压缩
        enhanced_features = self.reduce_conv(enhanced_features)
        enhanced_features = enhanced_features.permute(0, 2, 1, 3, 4)

        # 门控机制
        if self.use_gate:
            gate = self.gate_net(
                torch.cat([curr_frame.mean(dim=1), enhanced_features.mean(dim=1)], dim=1)
            )
            gate = gate.unsqueeze(1)
            output = curr_frame * (1 - gate) + enhanced_features * gate
        else:
            output = enhanced_features

        # 更新历史队列
        self.history_queue.pop(0)
        self.history_queue.append(curr_frame.clone().detach())

        return output


class ChannelAttention3D(nn.Module):
    """3D通道注意力模块"""

    def __init__(self, embed_dims, norm_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)

        self.conv1 = nn.Sequential(
            nn.Conv3d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, embed_dims),
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(embed_dims, embed_dims // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(embed_dims // 8, embed_dims, kernel_size=1),
            nn.GELU(),
        )

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv3d(embed_dims, embed_dims, kernel_size=3, padding=1),
            nn.GELU(),
            build_norm_layer(norm_cfg, embed_dims)[1]
        )

        self.layer_scale = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        x = self.conv1(input)
        y = self.sigmoid(self.conv2(self.avg_pool(x)))
        out = y * x
        out = self.conv(out)
        return input + self.layer_scale * out