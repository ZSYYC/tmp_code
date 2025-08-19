import logging
import os
import json
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Make sure these imports point to the correct file locations in your project
from .backbones.internvideo2 import InternVideo2
from .backbones.internvideo2 import TextTransformer, ClipTokenizer

logger = logging.getLogger(__name__)


class InternVideo2_CLIP_small_vision_only(nn.Module):
    """
    This class loads and uses only the vision part of the InternVideo2_CLIP_small model.
    """
    def __init__(self, config, tokenizer=None, is_pretrain=True):
        super().__init__()

        self.config = config
        self.is_pretrain = is_pretrain

        # 1. Build Vision Components
        self.vision_encoder = self.build_vision_encoder()
        self.vision_align = nn.Sequential(
            nn.LayerNorm(self.config.model.vision_encoder.clip_embed_dim),
            nn.Linear(
                self.config.model.vision_encoder.clip_embed_dim, 
                self.config.model.vision_encoder.align_dim
            ),
        )

        # 2. Define Image/Video Pre-processing Transform
        img_size = self.config.model.vision_encoder.img_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.Lambda(lambda x: x.float().div(255.0)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        
        # 3. Load weights from the unified checkpoint
        # NOTE: We assume the user provides a single checkpoint path in the config.
        # For consistency with the original code, we read from 'vision_ckpt_path'.
        ckpt_path = config.model.vision_ckpt_path
        if ckpt_path:
            self.load_from_checkpoint(ckpt_path)
        else:
            logger.warning("No checkpoint path provided for the vision model.")

    def build_vision_encoder(self):
        """Builds the InternVideo2 vision encoder."""
        vision_encoder = InternVideo2(
            in_chans=self.config.model.vision_encoder.in_chans,
            patch_size=self.config.model.vision_encoder.patch_size,
            img_size=self.config.model.vision_encoder.img_size,
            qkv_bias=self.config.model.vision_encoder.qkv_bias,
            drop_path_rate=self.config.model.vision_encoder.drop_path_rate,
            head_drop_path_rate=self.config.model.vision_encoder.head_drop_path_rate,
            embed_dim=self.config.model.vision_encoder.embed_dim,
            num_heads=self.config.model.vision_encoder.num_heads,
            mlp_ratio=self.config.model.vision_encoder.mlp_ratio,
            init_values=self.config.model.vision_encoder.init_values,
            qk_normalization=self.config.model.vision_encoder.qk_normalization,
            depth=self.config.model.vision_encoder.depth,
            use_flash_attn=self.config.model.vision_encoder.use_flash_attn,
            use_fused_rmsnorm=self.config.model.vision_encoder.use_fused_rmsnorm,
            use_fused_mlp=self.config.model.vision_encoder.use_fused_mlp,
            fused_mlp_heuristic=self.config.model.vision_encoder.fused_mlp_heuristic,
            attn_pool_num_heads=self.config.model.vision_encoder.attn_pool_num_heads,
            clip_embed_dim=self.config.model.vision_encoder.clip_embed_dim,
            layerscale_no_force_fp32=self.config.model.vision_encoder.layerscale_no_force_fp32,
            num_frames=self.config.model.vision_encoder.num_frames,
            tubelet_size=self.config.model.vision_encoder.tubelet_size,
            sep_pos_embed=self.config.model.vision_encoder.sep_pos_embed,
            use_checkpoint=self.config.model.vision_encoder.use_checkpoint,
            checkpoint_num=self.config.model.vision_encoder.checkpoint_num,
        )
        return vision_encoder

    def load_from_checkpoint(self, ckpt_path):
        """Loads vision-related weights from a single, unified checkpoint file."""
        logger.info(f"Loading vision weights from unified checkpoint: {ckpt_path}")
        full_ckpt = torch.load(ckpt_path, map_location='cpu')

        # Handle checkpoints saved with DDP or under a 'model' key
        if 'module' in full_ckpt:
            full_ckpt = full_ckpt['module']
        elif 'model' in full_ckpt:
            full_ckpt = full_ckpt['model']

        vision_state_dict = {}
        for k, v in full_ckpt.items():
            if k.startswith('vision_encoder.') or k.startswith('vision_align.'):
                vision_state_dict[k] = v
        
        msg = self.load_state_dict(vision_state_dict, strict=False)
        logger.info(f"Vision model weight loading message: {msg}")

    def encode_vision(self, image):
        """Encodes an image or video tensor into features."""
        T = image.shape[1]
        use_image = True if T == 1 else False
        # [B,T,C,H,W] -> [B,C,T,H,W]
        image = image.permute(0, 2, 1, 3, 4) 

        vision_embeds = self.vision_encoder(image, use_image=use_image)
        vision_embeds = self.vision_align(vision_embeds)
        return vision_embeds

    def get_vid_feat(self, frames: torch.Tensor):
        """
        A high-level function to get normalized video features for given frames.
        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].
        Returns:
            vfeat (torch.Tensor): The normalized output features. Shape: [B, C].
        """
        self.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            vfeat = self.encode_vision(frames)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vfeat