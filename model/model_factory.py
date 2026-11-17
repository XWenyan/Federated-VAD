import torch
from model.fed_baseline_mae_model import FedBaselineMaskedAutoencoderViT

def mae_cvt_patch16(args, **kwargs):
    model = FedBaselineMaskedAutoencoderViT(
            patch_size=16,
            in_chans=9,
            out_chans=4,
            img_size=args.input_size,
            embed_dim=256,
            depth=3,
            num_heads=4,
            decoder_embed_dim=128,
            decoder_depth=3,
            decoder_num_heads=4,
            mlp_ratio=2,
            norm_layer=torch.nn.LayerNorm,
            norm_pix_loss=args.norm_pix_loss,
            use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
            abnormal_score_func=args.abnormal_score_func,
            masking_method=args.masking_method,
            grad_weighted_loss=args.grad_weighted_rec_loss
        ).float()
    return model

def mae_cvt_patch8(args, **kwargs):
    model = FedBaselineMaskedAutoencoderViT(
            patch_size=8,
            in_chans=9,
            out_chans=4,
            img_size=args.input_size,
            embed_dim=256,
            depth=3,
            num_heads=4,
            decoder_embed_dim=128,
            decoder_depth=3,
            decoder_num_heads=4,
            mlp_ratio=2,
            norm_layer=torch.nn.LayerNorm,
            norm_pix_loss=args.norm_pix_loss,
            use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
            abnormal_score_func=args.abnormal_score_func,
            masking_method=args.masking_method,
            grad_weighted_loss=args.grad_weighted_rec_loss
        ).float()
    return model