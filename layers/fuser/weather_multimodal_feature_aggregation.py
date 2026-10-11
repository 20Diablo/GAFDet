import torch
from torch import nn
import torch.nn.functional as F

from mmcv.runner.base_module import ModuleList
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_positional_encoding
from mmcv.runner import auto_fp16

from ..modules.multimodal_deformable_cross_attention import DeformableCrossAttention

# 天气感知的自适应交叉注意力模块 Weather-aware Adaptive Cross-Attention
class WACA(nn.Module):
    def __init__(self, img_dims=80, radar_dims=80, weather_classes=8, reduction=4):
        super().__init__()

        # 增强版天气分类网络
        self.weather_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 80, 1, 1]
            nn.Flatten(),  # [B, 80]
            nn.Linear(80, 128),  # 调整为更合适的神经元数量
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, weather_classes)  # [B, 8]
        )

        # 联合权重生成MLP
        self.weight_mlp = nn.Sequential(
            nn.Linear(weather_classes + 1, 64),  # 天气类别 + 相似度，调整中间层大小
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )


        # 添加余弦相似度函数
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)


    def compute_similarity(self, img_feat, radar_feat):
        """计算图像特征与雷达特征之间的零均值余弦相似度

        Args:
            img_feat: 图像特征, shape=[B, 80, 128, 128]
            radar_feat: 雷达特征, shape=[B, 80, 128, 128]

        Returns:
            similarity: 特征相似度, shape=[B, 1]
        """
        # 计算并减去均值
        img_mean = img_feat.mean(1, keepdim=True)  # [B, 1, 128, 128]
        radar_mean = radar_feat.mean(1, keepdim=True)  # [B, 1, 128, 128]

        img_feat_zero_mean = img_feat - img_mean
        radar_feat_zero_mean = radar_feat - radar_mean

        # 计算余弦相似度
        similarity = self.cossim(img_feat_zero_mean, radar_feat_zero_mean)  # [B, 128, 128]

        # 全局平均池化得到单一相似度值
        similarity = similarity.mean([1, 2]).unsqueeze(1)  # [B, 1]

        return similarity

    def forward(self, img_feat, radar_feat):
        """
        前向传播函数

        Args:
            img_feat: 图像特征, shape=[B, 80, 128, 128]
            radar_feat: 雷达特征, shape=[B, 80, 128, 128]

        Returns:
            fused_img: 融合后的图像特征, shape=[B, 80, 128, 128]
            fused_radar: 融合后的雷达特征, shape=[B, 80, 128, 128]
        """

        # 天气信息提取
        weather_logits = self.weather_net(img_feat)  # [B, 8]
        weather_probs = F.softmax(weather_logits, dim=1)

        # 跨模态相似度计算
        similarity = self.compute_similarity(img_feat, radar_feat)  # [B, 1]

        # 联合权重生成
        joint_input = torch.cat([weather_probs, similarity], dim=1)  # [B, 9] (8类天气+1相似度)
        weights = self.weight_mlp(joint_input)  # [B, 2]

        # 权重应用（带残差）- 确保BCHW输出格式
        img_weight = weights[:, 0].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        radar_weight = weights[:, 1].view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        # 使用加权和进行特征融合，保持BCHW格式
        fused_img = img_feat * img_weight + img_feat  # [B, 80, 128, 128]
        fused_radar = radar_feat * radar_weight + radar_feat  # [B, 80, 128, 128]

        return fused_img, fused_radar


class MFAFuser(nn.Module):
    def __init__(self, num_sweeps=4, img_dims=80, pts_dims=128, embed_dims=256,
                 num_layers=6, num_heads=4, bev_shape=(128, 128)):
        super(MFAFuser, self).__init__()

        self.num_modalities = 2
        self.use_cams_embeds = False

        self.num_heads = num_heads

        self.img_dims = img_dims
        self.pts_dims = pts_dims
        self.embed_dims = embed_dims
        _pos_dim_ = self.embed_dims//2
        _ffn_dim_ = self.embed_dims*2

        self.norm_img = build_norm_layer(dict(type='LN'), img_dims)[1]
        self.norm_pts = build_norm_layer(dict(type='LN'), pts_dims)[1]
        self.input_proj = nn.Linear(img_dims + pts_dims, self.embed_dims)

        self.bev_h, self.bev_w = bev_shape

        self.positional_encoding = build_positional_encoding(
            dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=self.bev_h,
                col_num_embed=self.bev_w,
            ),
        )
        self.register_buffer('ref_2d', self.get_reference_points(self.bev_h, self.bev_w))

        ffn_cfgs = dict(
            type='FFN',
            embed_dims=self.embed_dims,
            feedforward_channels=_ffn_dim_,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
        )
        norm_cfgs = dict(type='LN')

        self.ffn_layers = ModuleList()
        for _ in range(num_layers):
            self.ffn_layers.append(
                build_feedforward_network(ffn_cfgs)
            )
        self.norm_layers1 = ModuleList()
        for _ in range(num_layers):
            self.norm_layers1.append(
                build_norm_layer(norm_cfgs, self.embed_dims)[1],
            )
        self.norm_layers2 = ModuleList()
        for _ in range(num_layers):
            self.norm_layers2.append(
                build_norm_layer(norm_cfgs, self.embed_dims)[1],
            )
        self.attn_layers = ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                DeformableCrossAttention(
                    img_dims=self.img_dims,
                    pts_dims=self.pts_dims,
                    embed_dims=self.embed_dims,
                    num_heads=self.num_heads,
                    num_modalities=self.num_modalities,
                    num_points=4
                ),
            )

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(embed_dims*num_sweeps,
                      embed_dims,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )

        #添加特征适配器
        self.feat_adaptor = WACA(
            img_dims=img_dims,
            radar_dims=pts_dims
        )

        self.init_weights()



    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 初始化WACA内部Linear层
        for m in self.modules():
            if isinstance(m, WACA):
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Linear):
                        nn.init.kaiming_uniform_(sub_m.weight, a=0, mode='fan_in')
                        if sub_m.bias is not None:
                            nn.init.constant_(sub_m.bias, 0)

        for m in self.modules():
            if isinstance(m, DeformableCrossAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @staticmethod
    def get_reference_points(H, W, dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.unsqueeze(2).unsqueeze(3)
        return ref_2d

    @auto_fp16(apply_to=('feat_img', 'feat_pts'))
    def _forward_single_sweep(self, feat_img, feat_pts):
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        bs = feat_img.shape[0]
        ref_2d_stack = self.ref_2d.repeat(bs, 1, 1, self.num_modalities, 1)

        feat_img, feat_pts = self.feat_adaptor(feat_img, feat_pts)

        feat_img = self.norm_img(feat_img.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        feat_pts = self.norm_pts(feat_pts.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        feat_flatten = []
        spatial_shapes = []
        for feat in [feat_img, feat_pts]:
            _, _, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(0, 2, 1).contiguous()  # [bs, num_cam, c, dw] -> [num_cam, bs, dw, c]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_img.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        bev_queries = torch.cat(feat_flatten, -1)
        bev_queries = self.input_proj(bev_queries)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(feat_img.dtype)
        bev_pos = self.positional_encoding(bev_mask).to(feat_img.dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()

        feat_img = feat_flatten[0]
        feat_pts = feat_flatten[1]
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['fusion_pre'].append(t1.elapsed_time(t2))

        for attn_layer, ffn_layer, norm_layer1, norm_layer2 in \
            zip(self.attn_layers, self.ffn_layers, self.norm_layers1, self.norm_layers2):
            # post norm
            bev_queries = attn_layer(
                bev_queries,
                feat_img,
                feat_pts,
                identity=None,
                query_pos=bev_pos,
                reference_points=ref_2d_stack,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                )
            bev_queries = norm_layer1(bev_queries)
            bev_queries = ffn_layer(bev_queries, identity=None)
            bev_queries = norm_layer2(bev_queries)
        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['fusion_layer'].append(t2.elapsed_time(t3))

        output = bev_queries.permute(0, 2, 1).contiguous().reshape(bs, self.embed_dims, h, w)
        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['fusion_post'].append(t3.elapsed_time(t4))

        return output

    def forward(self, feats, times=None):
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        num_sweeps = feats.shape[1]
        key_frame_res = self._forward_single_sweep(
            feats[:, 0, :self.img_dims],
            feats[:, 0, self.img_dims:self.img_dims+self.pts_dims]
        )
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['fusion'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            return key_frame_res, self.times

        ret_feature_list = [key_frame_res]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    feats[:, sweep_index, :self.img_dims],
                    feats[:, sweep_index, self.img_dims:self.img_dims+self.pts_dims])
                ret_feature_list.append(feature_map)

        return self.reduce_conv(torch.cat(ret_feature_list, 1)).float(), self.times