import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from model.cvt import ConvEmbed, Block
from util.morphology import Erosion2d, Dilation2d


class MaskedAutoencoderCvT(nn.Module):
    def __init__(self, img_size=(512,512), patch_size=16, in_chans=9, out_chans=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_only_masked_tokens_ab=False, abnormal_score_func='L1', masking_method="random_masking",
                 grad_weighted_loss=True):
        super().__init__()
        # --------------------------------------------------------------------------
        # Abnormal specifics
        self.use_only_masked_tokens_ab = use_only_masked_tokens_ab # 选择掩码策略(如随机掩码、梯度引导掩码)
        self.abnormal_score_func = abnormal_score_func[0] # 是否使用梯度加权损失
        # --------------------------------------------------------------------------

        self.masking = getattr(self, masking_method) # 选择掩码策略
        self.grad_weighted_loss=grad_weighted_loss # 是否使用梯度加权损失

        # --------------------------------------------------------------------------
        # MAE encoder specifics MAE编码器组件
        self.patch_embed = ConvEmbed( # 卷积嵌入模块：将图像分块并通过卷积映射到高维特征
            # img_size=img_size,
            patch_size=patch_size, # patch大小
            in_chans=in_chans, # 输入通道数(如9通道多模态数据)
            stride=patch_size, # 步长=patch大小，确保无重叠分块
            padding=0,
            embed_dim=embed_dim, # 嵌入维度
            norm_layer=norm_layer
        )
        self.patch_size = patch_size # 保存patch大小
        self.num_patches = img_size[0]//patch_size*img_size[1]//patch_size # 计算总patch数量：图像尺寸/patch尺寸 后相乘
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )
        # 编码器Transformer块：由多个Block组成
        self.blocks = nn.ModuleList([
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) # 编码器输出归一化
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics 解码器
        # 将编码器的输出映射到解码器维度
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # 掩码token：用于填充被掩码的位置
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 解码器Transformer块
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim) # @解码器输出归一化
        # 解码器预测头：将解码器输出映射为patch的像素值(patch_size^2 * 输出通道数)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)  # decoder to patch

        self.out_chans=out_chans # 输出通道数
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss # 是否对像素进行归一化后计算损失
        # 形态学操作：用于后续可能的掩码或异常图处理
        self.erosion = Erosion2d(1, 1, 2, soft_max=False)
        self.dilation = Dilation2d(1, 1, 3, soft_max=False)

        self.erosion_3 = Erosion2d(3, 3, 2, soft_max=False)
        self.dilation_3 = Dilation2d(3, 3, 3, soft_max=False)

    def freeze_backbone(self):
        """冻结骨干网络参数(用于微调阶段，固定预训练特征提取部分)"""
        self.cls_token.requires_grad = False
        self.mask_token.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for param in self.decoder_norm.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.decoder_embed.parameters():
            param.requires_grad = False
        for param in self.decoder_pred.parameters():
            param.requires_grad = False
        for i in range(0, len(self.decoder_blocks)):
            for param in self.decoder_blocks[i].parameters():
                param.requires_grad = False

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        """
        将图像分割为patch并展平
        imgs: 输入图像,形状为 (N, out_chans, H, W)
        return: 展平后的patch序列,形状为 (N, L, patch_size^2 * out_chans),其中L为patch数量
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 # 确保图像尺寸可被patch大小整除

        h = imgs.shape[2] // p # 高度方向的patch数量
        w = imgs.shape[3] // p # 宽度方向的patch数量

        # 维度重排：将每个patch的像素展平
        x = imgs.reshape(shape=(imgs.shape[0], self.out_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x) # 调整维度顺序
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.out_chans)) # 展平为序列
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        """
        将预测的patch序列重组为图像
        x: 预测的patch序列,形状为 (N, L, patch_size^2 * out_chans)
        return: 重组后的图像,形状为 (N, out_chans, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = 20 # 高度方向的patch数量(根据实际图像尺寸硬编码，可优化)
        w=40 # 宽度方向的patch数量(同上)
        assert h * w == x.shape[1]

        # 维度重排：将展平的patch恢复为空间结构
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, grad_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        """
        随机掩码策略:随机选择部分patch进行掩码(被掩码的patch将被模型重建)
        x: 输入特征，形状为 (N, D, H, W)(D为嵌入维度)
        mask_ratio: 掩码比例(如0.75表示掩码75%的patch)
        grad_mask: 梯度掩码(此处未使用，仅为接口统一)
        return:
            x_masked: 保留的patch特征,形状为 (N, len_keep, D)
            mask: 掩码标记(1表示被掩码,0表示保留)，形状为 (N, L)
            ids_restore: 用于恢复原始顺序的索引，形状为 (N, L)
        """
        N, D, H, W = x.shape  # batch, length, dim
        L = H*W # 总patch数量
        x = rearrange(x, 'b c h w -> b (h w) c') # 展平为序列 (N, L, D)
        len_keep = int(L * (1 - mask_ratio)) # 保留的patch数量

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] # 生成随机噪声（用于排序选择保留的patch）

        # sort noise for each sample 对噪声排序，获取打乱的索引（小值对应保留的patch）
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset 保留前len_keep个patch
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove # 生成掩码（1表示被掩码，0表示保留）
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # 保存掩码后的特征图尺寸
        self.masked_H = H
        self.masked_W = int(W*(1.-mask_ratio))
        self.H = H
        self.W = W
        return x_masked, mask, ids_restore

    def grad_masking_v1(self, x, mask_ratio, grad_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        """
        梯度引导掩码策略:基于梯度信息选择掩码的patch(优先掩码梯度小的区域，即模型已熟悉的区域）
        逻辑与random_masking类似,但使用梯度掩码代替随机噪声进行排序
        """
        grad_mask = F.max_pool2d(grad_mask, self.patch_size).max(1).values
        grad_mask = rearrange(grad_mask, 'b h w -> b (h w)')

        N, D, H, W = x.shape  # batch, length, dim
        L = H*W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))

        # sort noise for each sample 基于梯度掩码排序
        ids_shuffle = torch.argsort(grad_mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.masked_H = H
        self.masked_W = int(W*(1.-mask_ratio))
        self.H = H
        self.W = W
        # x_masked = rearrange(x_masked, 'b (h w) c -> b c h w', h=self.masked_H, w=self.masked_W)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, grad_mask):
        """
        编码器前向传播：将输入图像转换为特征，并进行掩码
        x: 输入图像，形状为 (N, in_chans, H, W)
        mask_ratio: 掩码比例
        grad_mask: 梯度掩码（用于梯度引导掩码策略）
        return:
            x: 编码器输出特征，形状为 (N, len_keep + 1, embed_dim)（+1是cls_token）
            mask: 掩码标记
            ids_restore: 恢复索引
        """
        # embed patches 1. patch嵌入：将图像转换为patch特征序列
        x = self.patch_embed(x)

        # masking: length -> length * mask_ratio 2. 掩码操作：保留部分patch，掩码其余部分
        x, mask, ids_restore = self.masking(x, mask_ratio, grad_mask)

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks 通过Transformer块提取特征
        for blk in self.blocks:
            x = blk(x, self.masked_H, self.masked_W)
        x = self.norm(x) # 归一化

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        解码器前向传播:基于编码器输出和掩码信息,重建被掩码的patch
        x: 编码器输出特征，形状为 (N, len_keep + 1, embed_dim)
        ids_restore: 恢复索引(用于将保留的patch放回原始位置)
        return: 重建的patch序列,形状为 (N, L, patch_size^2 * out_chans)
        """
        # embed tokens 将编码器特征映射到解码器维度
        x = self.decoder_embed(x)

        # 2. 添加掩码token：填充被掩码的位置
        # append mask tokens to sequence
        # 计算需要补充的掩码token数量
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # 3. 通过解码器Transformer块处理
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, self.H, self.W)
        x = self.decoder_norm(x)

        # predictor projection 预测patch的值
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


    def forward_loss(self, imgs, gradients, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        """
        计算损失函数：衡量重建结果与真实图像的差异
        imgs: 真实图像（目标），形状为 (N, out_chans, H, W)
        gradients: 梯度信息（用于加权损失）
        pred: 重建的patch序列,形状为 (N, L, patch_size^2 * out_chans)
        mask: 掩码标记
        return: 计算得到的损失值
        """
        target = self.patchify(imgs) # 将真实图像转换为patch序列（与pred格式匹配）
        # 若启用像素归一化，则对目标patch进行标准化
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        # 计算MSE损失（每个patch的平均损失）
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        min_magnitude_anomaly = torch.ones((gradients.shape[0],1,1,1), device=imgs.device) * 128
        # 梯度加权损失：对异常区域（梯度大的区域）赋予更高权重
        if self.grad_weighted_loss:
            anomaly_map = imgs[:, 3:, :, :]
            anomaly_map = torch.clip(anomaly_map, min=0, max=1)
            # 结合梯度信息调整异常图权重
            anomaly_map *= torch.maximum(min_magnitude_anomaly, torch.amax(gradients, dim=(1, 2, 3), keepdim=True))
            gradients += anomaly_map # 梯度融合异常信息
            
            # 下采样梯度到patch级别，作为损失权重
            grad_weights = F.max_pool2d(gradients, self.patch_size).mean(1)
            grad_weights = rearrange(grad_weights, 'b h w -> b (h w)')
            grad_weights = grad_weights / grad_weights.sum(dim=1, keepdims=True)
            loss = (loss * grad_weights).sum() # 加权求和损失
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self, imgs, targets, grad_mask=None, mask_ratio=0.75):
        """
        模型整体前向传播
        imgs: 输入图像（待编码），形状为 (N, in_chans, H, W)
        targets: 目标图像（待重建），形状为 (N, out_chans, H, W)
        grad_mask: 梯度掩码（用于梯度引导掩码）
        mask_ratio: 掩码比例
        return:
            训练时：损失值、重建结果、掩码
            测试时：损失值、重建结果、掩码、异常分数
        """
        # 1. 编码器：获取掩码后的特征、掩码标记、恢复索引
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, grad_mask)
        # 2. 解码器：重建被掩码的patch
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # 3. 计算损失
        loss = self.forward_loss(targets, grad_mask, pred, mask)
        # 4. 训练/测试模式返回不同结果
        if self.training:
            return loss, pred, mask
        else:
            return loss, pred, mask, self.abnormal_score(targets, pred, mask, grad_mask)

    def abnormal_score(self, imgs, pred, mask, gradients):
        imgs = self.patchify(imgs)
        if self.use_only_masked_tokens_ab:
            mask = mask.bool()
            selected_pred = []
            selected_lbl = []
            for i in range(0, imgs.shape[0]):
                selected_pred.append(pred[i][mask[i]])
                selected_lbl.append(imgs[i][mask[i]])

            pred = torch.stack(selected_pred)
            imgs = torch.stack(selected_lbl)
        return ((imgs - pred) ** 2).mean((1, 2))  # MSE

