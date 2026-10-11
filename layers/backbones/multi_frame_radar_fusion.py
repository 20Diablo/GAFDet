import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import build_norm_layer


class RadarHistoryFusion(nn.Module):
    def __init__(self, feature_dim=80, history_frames=3, use_gate=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.history_frames = history_frames
        self.history_queue = []
        self.use_gate = use_gate
        print(f"RadarHistoryFusion: history_frames={history_frames}")

        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

        #运动估计网络
        self.flow_estimation = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)  # 输出光流场 (vx, vy)
        )

        # 偏移量生成网络
        self.offset_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 72, kernel_size=3, padding=1)
        )
        # 可变形对齐模块
        self.deform_align = torchvision.ops.DeformConv2d(
            feature_dim, feature_dim, kernel_size=3, padding=1, groups=4
        )

        # 3D通道注意力融合模块
        self.channel_att = ChannelAttention3D(embed_dims=feature_dim * (history_frames + 1))

        # 通道压缩卷积（保持与原始结构兼容）
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
        """
        使用可变形卷积对齐历史帧特征
        输入:
            - hist_feat: 历史帧特征 [B, T, C, H, W]
            - flow: 光流场 [B, T, 2, H, W]
        输出:
            - 对齐后的历史帧特征 [B, T, C, H, W]
        """
        B, T, C, H, W = hist_feat.shape

        # 使用卷积生成偏移量
        offsets = self.offset_conv(flow.view(B * T, 2, H, W))  # [B*T, 72, H, W]
        offsets = offsets.view(B, T, 72, H, W)  # [B, T, 72, H, W]

        # 逐帧对齐
        aligned_feats = []
        for t in range(T):
            offset_t = offsets[:, t]  # [B, 72, H, W]
            aligned = self.deform_align(
                hist_feat[:, t].contiguous(),  # 输入特征 [B, C, H, W]
                offset_t  # 偏移量 [B, 72, H, W]
            )
            aligned_feats.append(aligned.unsqueeze(1))  # [B, 1, C, H, W]

        return torch.cat(aligned_feats, dim=1)  # [B, T, C, H, W]

    def compute_similarity(self, curr_feat, hist_feat):
        """计算当前帧与历史帧的余弦相似度"""
        # curr_feat和hist_feat的shape都是(B, num_sweep, C, H, W)
        B, T, C, H, W = curr_feat.shape

        # 将特征重塑为(B*T, C, H, W)用于计算均值
        curr_feat_reshaped = curr_feat.view(-1, C, H, W)
        hist_feat_reshaped = hist_feat.view(-1, C, H, W)

        # 计算并减去均值
        curr_mean = curr_feat_reshaped.mean(1, keepdim=True)  # (B*T, 1, H, W)
        hist_mean = hist_feat_reshaped.mean(1, keepdim=True)  # (B*T, 1, H, W)

        curr_feat_zero_mean = curr_feat_reshaped - curr_mean
        hist_feat_zero_mean = hist_feat_reshaped - hist_mean

        # 计算余弦相似度
        similarity = self.cossim(curr_feat_zero_mean, hist_feat_zero_mean)  # (B*T, H, W)

        # 重塑回原始维度并取平均
        return similarity.unsqueeze(1)

    def forward(self, x):
        """
        输入: x shape is (B, num_sweep, C, H, W)
        B: batch size
        num_sweep: 每帧中的扫描数量（这里是4）
        C: 特征维度
        H, W: BEV特征图大小
        """
        batch_size, num_sweep, C, H, W = x.shape
        assert C == self.feature_dim, f"Feature dimension mismatch: expected {self.feature_dim}, got {C}"

        # 对前几帧，直接返回原始特征
        if len(self.history_queue) < self.history_frames:
            self.history_queue.append(x.clone().detach())  # 存储当前帧（包含多个sweep）
            return x # 返回原始特征

        # 获取当前帧（包含多个sweep）
        curr_frame = x  # shape: (B, num_sweep, C, H, W)

        # 对每个历史帧进行运动对齐和相似度加权
        weighted_history_frames = []
        # 计算与历史帧的注意力权重并加权
        for hist_frame in self.history_queue:
            # 逐帧计算光流
            flows = []
            for t in range(num_sweep):
                # 计算当前帧和历史帧的光流
                flow = self.flow_estimation(
                    torch.cat([curr_frame[:, t], hist_frame[:, t]], dim=1)  # [B, 2*C, H, W]
                )
                flows.append(flow.unsqueeze(1))  # [B, 1, 2, H, W]

            # 合并光流
            flow = torch.cat(flows, dim=1)  # [B, T, 2, H, W]

            # 可变形对齐
            aligned_hist = self.align_features(hist_frame, flow)

            similarity = self.compute_similarity(curr_frame, aligned_hist)

            weighted_hist_frame = aligned_hist.view(-1, C, H, W) * similarity
            weighted_hist_frame = weighted_hist_frame.view(batch_size, num_sweep, C, H, W).contiguous()
            weighted_history_frames.append(weighted_hist_frame)

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
            output = curr_frame + enhanced_features * gate
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

