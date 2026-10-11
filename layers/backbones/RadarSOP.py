import torch
import torch.nn as nn
import torch.nn.functional as F


class RadarSOP(nn.Module):
    def __init__(self, in_channels=80, reduct_ratio=4, patch_size=8, pe_dim=8):
        super().__init__()


        self.patch_size = patch_size
        self.pe_dim = pe_dim

        # 轻量级特征压缩
        self.channel_reduct = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduct_ratio, 1),
            nn.BatchNorm2d(in_channels // reduct_ratio),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 分块协方差计算
        self.cov_proj = nn.Conv2d(in_channels // reduct_ratio, pe_dim, 3, padding=1)

        # SVD增强参数
        self.svd_fc = nn.Linear(pe_dim ** 2, patch_size ** 2, bias=False)

        # 特征重建
        self.reconstruct = nn.Sequential(
            nn.Conv2d(patch_size ** 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def _patch_svd(self, x):
        """分块SVD增强"""
        B, C, H, W = x.shape

        # 分块处理 [B, C, H, W] -> [B*num_patches, C, p, p]
        x = F.unfold(x, self.patch_size, stride=self.patch_size)  # [B, C*p*p, num_patches]
        x = x.view(B, C, self.patch_size ** 2, -1).permute(0, 3, 1, 2)  # [B, num_patches, C, p^2]
        x = x.reshape(-1, C, self.patch_size, self.patch_size)  # [B*num_patches, C, p, p]

        # 协方差投影
        cov = self.cov_proj(x)  # [B*num_patches, pe_dim, p, p]
        cov = cov.flatten(start_dim=2)  # [B*num_patches, pe_dim, p^2]

        cov = torch.bmm(cov, cov.transpose(1, 2))  # [B*num_patches, pe_dim, pe_dim]

        # 转换为全精度
        cov = cov.to(torch.float32)
        # 将协方差矩阵移至CPU计算SVD
        cov_cpu = cov.detach().cpu()  # 分离计算图并转CPU

        # 在CPU上进行SVD分解
        with torch.no_grad():
            U_cpu, S_cpu, V_cpu = torch.linalg.svd(cov_cpu, full_matrices=False)

        # 将结果移回GPU并保留梯度
        U = U_cpu.to(x.device).requires_grad_(True)
        S = S_cpu.to(x.device).requires_grad_(True)
        V = V_cpu.to(x.device).requires_grad_(True)
        S_sqrt = torch.sqrt(S + 1e-6)
        cov_sqrt = U * S_sqrt.unsqueeze(-1) @ V.transpose(1, 2)

        cov_flat = cov_sqrt.flatten(1)  # [B*num_patches, pe_dim^2]

        # 转回原始精度
        cov_flat = cov_flat.to(x.dtype)
        return self.svd_fc(cov_flat)  # [B*num_patches, p^2]

    def forward(self, x):
        """输入/输出形状: [B, C, H, W]"""
        identity = x

        # 特征压缩
        x_red = self.channel_reduct(x)  # [B, C//4, H, W]

        # 分块SVD增强
        B, C_red, H, W = x_red.shape
        svd_feat = self._patch_svd(x_red)  # [B*num_patches, p^2]

        # 重组特征图
        patches_h = H // self.patch_size
        patches_w = W // self.patch_size
        svd_feat = svd_feat.view(B, patches_h, patches_w, self.patch_size ** 2)
        svd_feat = svd_feat.permute(0, 3, 1, 2)  # [B, p^2, patches_h, patches_w]

        # 上采样回原始分辨率
        svd_feat = F.interpolate(
            svd_feat,
            scale_factor=self.patch_size,  # 直接按分块尺寸放大
            mode='bilinear',
            align_corners=False,
            recompute_scale_factor=True
        )

        # 重建原始维度
        out = self.reconstruct(svd_feat)  # [B, C, H, W]
        return out + identity


class MultiScaleRadarSOP(nn.Module):
    def __init__(self, in_channels=80):
        super().__init__()
        # 多尺度并行处理
        self.sop_global = RadarSOP(in_channels,
                                   patch_size=16,
                                   pe_dim=16)    # 大块捕获全局特征
        self.sop_local = RadarSOP(in_channels,
                                  patch_size=8,
                                  pe_dim=8)  # 小块捕获细节

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # 生成注意力权重
        )

    def forward(self, x):
        # 全局特征
        feat_global = self.sop_global(x)  # [B, C, H, W]

        # 局部特征
        feat_local = self.sop_local(x)  # [B, C, H, W]

        # 自适应融合
        fusion_weights = self.fusion(torch.cat([feat_global, feat_local], dim=1))
        return fusion_weights * feat_global + (1 - fusion_weights) * feat_local

if __name__ == "__main__":
    model = MultiScaleRadarSOP(in_channels=80)
    bev_input = torch.randn(4, 80, 128, 128)  # 模拟雷达BEV输入
    output = model(bev_input)  # 输出保持[4,80,128,128]
    print(output.shape)  # torch.Size([4, 80, 128, 128])`