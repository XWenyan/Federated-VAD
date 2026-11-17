# fed_baseline_mae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from functools import partial


class PatchEmbed(nn.Module):
    """2D图像到patch嵌入的转换模块"""
    def __init__(self, patch_size=16, in_chans=9, embed_dim=768, norm_layer=None):
        super().__init__()
        self.patch_size = self.to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=self.patch_size, 
                             stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    @staticmethod
    def to_2tuple(x):
        """将输入转换为二维元组"""
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                return (x[0], x[0])
            return x
        return (x, x)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"Input image size ({H}*{W}) is not divisible by patch size {self.patch_size}"
        
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class FedBaselineMaskedAutoencoderViT(nn.Module):
    """联邦学习基础版本的MAE模型 - 不使用scene_embed参数"""
    
    def __init__(self, patch_size=16, in_chans=9, out_chans=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_only_masked_tokens_ab=False, abnormal_score_func='L1', 
                 masking_method="random_masking", grad_weighted_loss=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = kwargs.get('img_size', (320, 640))
        
        # 异常检测相关设置
        self.use_only_masked_tokens_ab = use_only_masked_tokens_ab
        self.abnormal_score_func = abnormal_score_func[0] if abnormal_score_func else 'L'
        
        self.masking = getattr(self, masking_method)
        self.grad_weighted_loss = grad_weighted_loss
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.norm_pix_loss = norm_pix_loss

        # 计算patch网格尺寸
        self.grid_h = self.img_size[0] // patch_size
        self.grid_w = self.img_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # MAE编码器
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim, 
            norm_layer=norm_layer
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 编码器位置编码
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False
        )
        
        # 编码器Transformer块
        from timm.models.vision_transformer import Block
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # MAE解码器
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 解码器位置编码
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        # 解码器Transformer块
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        """初始化模型权重"""
        encoder_pos_embed = self.get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1], 
            self.grid_h, self.grid_w, cls_token=True
        )
        self.encoder_pos_embed.data.copy_(torch.from_numpy(encoder_pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = self.get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            self.grid_h, self.grid_w, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """辅助初始化函数"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_2d_sincos_pos_embed(self, embed_dim, grid_h, grid_w, cls_token=False):
        """生成2D正弦余弦位置嵌入"""
        grid_h_coords = np.arange(grid_h, dtype=np.float32)
        grid_w_coords = np.arange(grid_w, dtype=np.float32)
        grid = np.meshgrid(grid_w_coords, grid_h_coords)
        grid = np.stack(grid, axis=0)
        
        grid = grid.reshape([2, 1, grid_h, grid_w])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        """从网格坐标生成2D正弦余弦位置嵌入"""
        assert embed_dim % 2 == 0
        
        # 使用半维度进行正弦余弦编码
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """生成1D正弦余弦位置嵌入"""
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def patchify(self, imgs):
        """将图像转换为patch序列"""
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.out_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.out_chans))
        return x

    def unpatchify(self, x):
        """将patch序列还原为图像"""
        p = self.patch_size
        h = self.grid_h
        w = self.grid_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, grad_mask):
        """随机掩码patch"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成掩码 1表示被掩码
        mask = torch.ones([N, L], device=x.device) 
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, grad_mask):
        '''编码器前向传播'''
        # patch嵌入
        x = self.patch_embed(x)
        B, L, D = x.shape
        
        # 掩码操作
        x, mask, ids_restore = self.masking(x, mask_ratio, grad_mask)
        
        # 添加cls token和位置编码
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder_pos_embed[:, :x.shape[1], :]

        # 通过Transformer块提取特征
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        '''解码器前向传播 - 基础联邦学习版本，不接收scene_embed参数'''
        # 嵌入到解码器维度
        x = self.decoder_embed(x)
        
        # 计算需要添加的掩码token数量
        B = x.shape[0]
        num_original_patches = ids_restore.shape[1]
        num_current_patches = x.shape[1] - 1
        num_mask_tokens = num_original_patches - num_current_patches
        
        mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1)
        
        # 分离cls token和patch tokens
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        
        # 将可见patch和mask tokens拼接
        x_ = torch.cat([patch_tokens, mask_tokens], dim=1)
        # 恢复原始顺序
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # 重新添加cls token
        x = torch.cat([cls_token, x_], dim=1)
        # 添加解码器位置编码
        x = x + self.decoder_pos_embed
        
        # 应用Transformer块 - 不使用scene_embed参数
        for blk in self.decoder_blocks:
            x = blk(x)  # 关键修改：不传递scene_embed
            
        x = self.decoder_norm(x)

        # 预测器投影
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # 移除cls token

        return x

    def forward_loss(self, imgs, gradients, pred, mask):
        '''计算重建损失'''
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        if self.grad_weighted_loss:
            # 梯度加权：增强异常区域（梯度大的区域）的损失权重
            anomaly_map = imgs[:, 3:, :, :]
            anomaly_map = torch.clip(anomaly_map, min=0, max=1)
            anomaly_weight = 0.3
            min_magnitude_anomaly = torch.ones((gradients.shape[0], 1, 1, 1), device=imgs.device) * 128
            anomaly_map *= torch.maximum(min_magnitude_anomaly, torch.amax(gradients, dim=(1, 2, 3), keepdim=True))
            gradients += anomaly_map * anomaly_weight
            
            grad_weights = F.max_pool2d(gradients, self.patch_size).mean(1)
            grad_weights = rearrange(grad_weights, 'b h w -> b (h w)')
            grad_weights = grad_weights / grad_weights.sum(dim=1, keepdims=True)
            loss = (loss * grad_weights).sum()
        else:
            # 仅计算被掩码补丁的损失（标准MAE损失）
            loss = (loss * mask).sum() / mask.sum()
            
        return loss

    def forward(self, imgs, targets=None, grad_mask=None, mask_ratio=0.75, scene_embed=None):
        '''模型整体前向传播 - 兼容联邦学习的调用方式'''
        # 注意：我们接收scene_embed参数但不使用它，以保持接口兼容性
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, grad_mask)
        pred = self.forward_decoder(latent, ids_restore)  # 不传递scene_embed
        loss = self.forward_loss(targets, grad_mask, pred, mask)
        
        if self.training:
            return loss, pred, mask
        else:
            # 测试时计算异常分数
            return loss, pred, mask, self.abnormal_score(targets, pred, mask, grad_mask)

    def abnormal_score(self, imgs, pred, mask, gradients):
        '''计算异常分数'''
        imgs = self.patchify(imgs)
        if self.use_only_masked_tokens_ab:
            mask = mask.bool()
            selected_pred = []
            selected_lbl = []
            for i in range(imgs.shape[0]):
                selected_pred.append(pred[i][mask[i]])
                selected_lbl.append(imgs[i][mask[i]])
            pred = torch.stack(selected_pred)
            imgs = torch.stack(selected_lbl)
        
        if self.abnormal_score_func == 'L1':
            score = (imgs - pred).abs().mean((1, 2))
        else:
            score = ((imgs - pred) ** 2).mean((1, 2))
    
        return score