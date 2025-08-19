import torch
from PIL import Image
import numpy as np

# Assume 'config' is your loaded configuration object

# 1. Initialize both models
vision_model = InternVideo2_CLIP_small_vision_only(config=config, is_pretrain=False).to(config.device)
text_model = InternVideo2_CLIP_small_text_only(config=config, is_pretrain=False).to(config.device)

vision_model.eval()
text_model.eval()

# 2. Prepare inputs
# Example video frames (random tensor)
# Shape: [Batch, Time, Channels, Height, Width]
video_frames = torch.rand(1, 8, 3, 224, 224).to(config.device) 

# Example text labels
text_labels = ["a dog running on the beach", "a cat sitting on a sofa", "a person playing guitar"]

# 3. Get features from each model
video_features = vision_model.get_vid_feat(video_frames) # Shape: [1, 512]
text_features = text_model.get_txt_feat(text_labels)     # Shape: [3, 512]

# 4. Perform zero-shot classification (manually)
# The logic from the original `predict_label` is now done outside the models
video_features = video_features.float()
text_features = text_features.float()

similarity_scores = (100.0 * video_features @ text_features.T).softmax(dim=-1)

# Get top prediction
top_probs, top_indices = similarity_scores.topk(1, dim=-1)

print("--- Inference Results ---")
print(f"Video Features Shape: {video_features.shape}")
print(f"Text Features Shape: {text_features.shape}")
print(f"Similarity Scores: {similarity_scores}")
print(f"Top prediction: '{text_labels[top_indices[0][0]]}' with a probability of {top_probs[0][0]:.4f}")