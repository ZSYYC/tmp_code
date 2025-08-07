import torch
from collections import OrderedDict
import argparse
import numpy as np
import cv2  # Used for resizing frames
from decord import VideoReader, cpu

# 关键步骤 1: 导入timm的模型创建工厂
from timm.models import create_model 

# 关键步骤 2: 执行模型注册, 让timm认识我们的自定义模型
from models import *

# ===============================================================
#                视频预处理部分 (参考您的Dataset)
# ===============================================================

class VideoPreprocessor:
    """
    一个独立的视频预处理器，模拟了您在训练时使用的验证/测试数据增强流程。
    """
    def __init__(self, args):
        self.num_frames = args.num_frames
        self.input_size = args.input_size
        self.short_side_size = args.short_side_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _sample_frames(self, num_total_frames):
        """
        确定性地从视频中采样帧索引 (中心化采样)。
        这模仿了您代码中的 _get_seq_frames(..., clip_idx!=-1) 的逻辑。
        """
        seg_size = float(num_total_frames - 1) / self.num_frames
        indices = []
        for i in range(self.num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            # 从每个段的中心取一帧
            idx = (start + end) // 2
            indices.append(idx)
        return indices

    def _resize_and_crop(self, frames):
        """
        对帧进行Resize（短边缩放）和CenterCrop。
        """
        resized_frames = []
        for frame in frames: # frame is HWC, NumPy array
            h, w, _ = frame.shape
            
            # 1. Resize short side
            if h < w:
                new_h = self.short_side_size
                new_w = int((new_h / h) * w)
            else:
                new_w = self.short_side_size
                new_h = int((new_w / w) * h)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 2. Center crop
            h, w, _ = resized_frame.shape
            start_x = (w - self.input_size) // 2
            start_y = (h - self.input_size) // 2
            cropped_frame = resized_frame[start_y:start_y+self.input_size, start_x:start_x+self.input_size]
            
            resized_frames.append(cropped_frame)
            
        return np.stack(resized_frames) # T, H, W, C

    def __call__(self, video_path):
        """
        处理单个视频文件。
        """
        # 1. 使用decord加载视频
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
        except Exception as e:
            print(f"无法加载视频 {video_path}: {e}")
            return None

        # 2. 采样帧索引
        indices = self._sample_frames(total_frames)
        
        # 3. 读取帧
        frames = vr.get_batch(indices).asnumpy()  # 返回 (T, H, W, C) NumPy array

        # 4. 空间变换 (Resize & Crop)
        frames = self._resize_and_crop(frames) # T, H_crop, W_crop, C

        # 5. 转换为Tensor并进行归一化
        # T, H, W, C -> C, T, H, W
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        tensor /= 255.0  # 缩放到 [0, 1]

        # 6. 标准化
        tensor = (tensor - torch.tensor(self.mean).view(3, 1, 1, 1)) / torch.tensor(self.std).view(3, 1, 1, 1)

        return tensor


def build_inference_model(args):
    """根据参数创建模型"""
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        num_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        sep_pos_embed=args.sep_pos_embed
    )
    return model

def load_finetuned_checkpoint(model, checkpoint_path, model_key='model|module'):
    """加载微调后的模型权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"成功从 {checkpoint_path} 加载权重文件。")
    checkpoint_model = checkpoint.get(model_key.split('|')[0], checkpoint)
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint_model.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
            
    msg = model.load_state_dict(new_state_dict, strict=False)
    print("加载权重信息:", msg)
    return model

if __name__ == '__main__':
    # 使用argparse来管理参数，使其更灵活
    parser = argparse.ArgumentParser('InternVideo2 Inference Script')
    parser.add_argument('--video_path', type=str, default="/root/hmdb51_single_modality_dataset/videos/20_good_form_pullups_pullup_f_nm_np1_ri_goo_1.avi", help='Path to the input video file.')
    parser.add_argument('--checkpoint', type=str, default="/root/InternVideo/InternVideo2/single_modality/scripts/finetuning/full_tuning/k400/S14_ft_k710_ft_k400_f8/best_ckpt.bin", help='Path to the model checkpoint file.')
    parser.add_argument('--label_map', type=str, default="/root/hmdb51_single_modality_dataset/label_map_train.txt", help='Path to a class label map file (one class per line).')
    
    # --- 模型参数 (必须与微调训练时一致) ---
    parser.add_argument('--model', type=str, default='internvideo2_small_patch14_224')
    parser.add_argument('--nb_classes', type=int, default=400)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--tubelet_size', type=int, default=1)
    parser.add_argument('--sep_pos_embed', action='store_false', dest='sep_pos_embed', help='Use unified position embedding', default=False)
    
    cli_args = parser.parse_args()

    # --- 1. 创建模型和预处理器 ---
    print("正在创建模型和预处理器...")
    model = build_inference_model(cli_args)
    preprocessor = VideoPreprocessor(cli_args)
    print("模型和预处理器创建成功。")

    # --- 2. 加载微调后的权重 ---
    print(f"正在从 {cli_args.checkpoint} 加载权重...")
    model = load_finetuned_checkpoint(model, cli_args.checkpoint)

    # --- 3. 准备推理 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).half().eval()
    print(f"模型已移至 {device} 并设置为评估模式(半精度)。")

    # --- 4. 预处理真实视频 ---
    print(f"正在处理视频: {cli_args.video_path}")
    input_tensor = preprocessor(cli_args.video_path)
    
    if input_tensor is None:
        exit("视频处理失败，程序退出。")

    # 添加batch维度，并移动到正确的设备和数据类型
    input_tensor = input_tensor.unsqueeze(0).to(device).half()
    print(f"视频预处理完成，输入张量形状: {input_tensor.shape}")

    # --- 5. 执行推理 ---
    with torch.no_grad():
        output = model(input_tensor)

    # --- 6. 处理输出 ---
    output = output.float() # 转回float32以保证softmax数值稳定性
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # 获取Top-5预测
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    print(f"\n--- 推理结果 ---")
    
    # 加载标签映射
    class_labels = None
    if cli_args.label_map:
        try:
            with open(cli_args.label_map, 'r') as f:
                class_labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"警告: 找不到标签映射文件 {cli_args.label_map}")

    for i in range(top5_prob.size(0)):
        prob = top5_prob[i].item()
        idx = top5_indices[i].item()
        label = class_labels[idx] if class_labels else f"类别索引 {idx}"
        print(f"{i+1}. {label}: {prob:.4f}")


