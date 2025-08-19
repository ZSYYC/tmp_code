import logging
import os
import json
import torch
from torch import nn

# Make sure these imports point to the correct file locations in your project
from .backbones.internvideo2 import TextTransformer, ClipTokenizer

logger = logging.getLogger(__name__)


class InternVideo2_CLIP_small_text_only(nn.Module):
    """
    This class loads and uses only the text part of the InternVideo2_CLIP_small model.
    """
    def __init__(self, config, tokenizer=None, is_pretrain=True):
        super().__init__()

        self.config = config
        self.is_pretrain = is_pretrain

        # 1. Build Text Components
        text_encoder_cfg = json.load(
            open(os.path.join(
                "./models/backbones/internvideo2/mobileclip/configs/",
                f"{self.config.model.text_encoder.name}.json"))
        )
        
        if tokenizer is None:
            self.tokenizer = ClipTokenizer(text_encoder_cfg)
        else:
            self.tokenizer = tokenizer
            
        self.text_encoder = self.build_text_encoder(
            cfg=text_encoder_cfg['text_cfg'], 
            projection_dim=text_encoder_cfg["embed_dim"]
        )
        
        # 2. Load weights from the unified checkpoint
        # NOTE: We assume the user provides a single checkpoint path in the config.
        # For consistency with the original code, we read from 'text_ckpt_path'.
        ckpt_path = config.model.text_ckpt_path
        if ckpt_path:
            self.load_from_checkpoint(ckpt_path)
        else:
            logger.warning("No checkpoint path provided for the text model.")


    def build_text_encoder(self, cfg, projection_dim):
        """Builds the TextTransformer text encoder."""
        text_encoder = TextTransformer(cfg, projection_dim)
        return text_encoder

    def load_from_checkpoint(self, ckpt_path):
        """Loads text-related weights from a single, unified checkpoint file."""
        logger.info(f"Loading text weights from unified checkpoint: {ckpt_path}")
        full_ckpt = torch.load(ckpt_path, map_location='cpu')

        # Handle checkpoints saved with DDP or under a 'model' key
        if 'module' in full_ckpt:
            full_ckpt = full_ckpt['module']
        elif 'model' in full_ckpt:
            full_ckpt = full_ckpt['model']

        text_state_dict = {}
        for k, v in full_ckpt.items():
            if k.startswith('text_encoder.'):
                text_state_dict[k] = v
        
        msg = self.load_state_dict(text_state_dict, strict=False)
        logger.info(f"Text model weight loading message: {msg}")
    
    def encode_text(self, text):
        """Encodes tokenized text into features."""
        text_embeds = self.text_encoder(text)
        return text_embeds
        
    def get_txt_feat(self, text: str):
        """
        A high-level function to get normalized text features for a given string.
        Args:
            text (str or list[str]): The input text prompt(s).
        Returns:
            tfeat (torch.Tensor): The normalized output features. Shape: [B, C].
        """
        self.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            tokenized_text = self.tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=self.config.max_txt_l, 
                return_tensors="pt",
            ).to(self.config.device)
            
            tfeat = self.encode_text(tokenized_text)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        return tfeat