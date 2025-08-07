import random
import numpy as np
import os
import io
import cv2
import torch
from tqdm import tqdm
from loguru import logger

# --- 从您的示例中导入所需的工具函数和配置 ---
from demo_config import (Config,
                         eval_dict_leaf)
from demo.utils import (retrieve_text_with_saved_features,
                      _frame_from_video,
                      setup_internvideo2,
                      save_text_features)
from dataset.text_prompt import kinetics_templates, kinetics_templates_action_clip
# --- 1. 设置路径和配置 ---
# TODO: 为什么这个评估代码效果要比直接使用评估脚本的评估效果低
# TODO: 将此路径更改为您的数据集根目录
# 预期结构:
# DATASET_ROOT/
#  ├── class_A/
#  │   ├── video1.mp4
#  │   └── video2.avi
#  ├── class_B/
#  │   ├── video3.mp4
#  └── ...
DATASET_ROOT = '/root/hmdb51' 
use_prompt = True  # 是否使用文本提示
# 如果使用提示，确保 kinetics_templates 已经定义并包含所需的模板

# 文本特征缓存文件的保存路径
SAVE_PATH = 'text_feature_cache/classification_text_features.npy'
# 确保缓存目录存在
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 加载模型配置
logger.info("Loading model configuration...")
config = Config.from_file('/root/InternVideo/InternVideo2/multi_modality/scripts/evaluation/clip/zero_shot/S14/config_hmdb51.py')
# config = Config.from_file('/root/InternVideo/InternVideo2/multi_modality/scripts/pretraining/clip/L14/config.py')
config = eval_dict_leaf(config)

# --- 2. 初始化模型 ---
logger.info("Setting up InternVideo2 model...")
intern_model, tokenizer = setup_internvideo2(config)
logger.info("Model setup complete.")

# --- 3. 收集所有类别并预计算文本特征 (如果需要) ---
logger.info(f"Checking for dataset at: {DATASET_ROOT}")
if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(f"Dataset root directory not found at: {DATASET_ROOT}")

# 将子文件夹名作为 captions  注意文件夹名的下划线到底要不要替换成空格，实际测试保持下划线的准确率高一些？？？？
captions_ori = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
if use_prompt:
    captions = [random.choice(kinetics_templates_action_clip).format(d.replace('_', ' ')) for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
else:
    captions = [d.replace('_', ' ') for d in captions_ori]  # 替换下划线为空格
if not captions:
    raise ValueError(f"No class subfolders found in {DATASET_ROOT}")
logger.info(f"Found {len(captions)} classes: {captions}")

# 如果文本特征没有被缓存，则计算并保存它们
if not os.path.exists(SAVE_PATH):
    logger.info(f"Cache not found. Computing and saving text features to {SAVE_PATH}...")
    save_text_features(captions, intern_model, save_path=SAVE_PATH)
    logger.info("Text features saved.")
else:
    logger.info(f"Loading pre-computed text features from {SAVE_PATH}.")

# 加载预计算的文本特征
saved_text_features = torch.load(SAVE_PATH, map_location='cpu')
# saved_text_features = np.load(SAVE_PATH, allow_pickle=True).item()
logger.info("Text features loaded.")


# --- 4. 执行评估 ---
r1_correct = 0
r3_correct = 0
r5_correct = 0
total_videos = 0

# 遍历每个类别文件夹
for idx, ground_truth_label in enumerate(tqdm(captions_ori, desc="Evaluating Classes", total=len(captions_ori))):
    class_folder_path = os.path.join(DATASET_ROOT, ground_truth_label)
    
    # 遍历该类别下的所有视频文件
    video_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    for video_name in video_files:
        video_path = os.path.join(class_folder_path, video_name)
        
        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                logger.warning(f"Skipping: Could not open video file {video_path}")
                continue
            
            # 从视频中提取帧
            frames = [x for x in _frame_from_video(video)]
            video.release()

            if not frames:
                logger.warning(f"Skipping: No frames extracted from {video_path}")
                continue

            # 使用预计算的特征进行文本检索
            # texts 是返回的 top-k 标签列表
            texts, probs = retrieve_text_with_saved_features(
                frames,
                saved_text_features,
                model=intern_model,
                topk=5, # 我们需要 top 5 来计算 R@5
                config=config
            )
            
            # 更新统计数据
            total_videos += 1
            
            # 检查 R@1
            if captions[idx] in texts[:1]:
                r1_correct += 1
            
            # 检查 R@3
            if captions[idx] in texts[:3]:
                r3_correct += 1
                
            # 检查 R@5
            if captions[idx] in texts[:5]:
                r5_correct += 1

        except Exception as e:
            logger.error(f"An error occurred while processing {video_path}: {e}")
            continue

# --- 5. 计算并打印结果 ---
if total_videos == 0:
    logger.error("No videos were processed. Cannot calculate metrics. Please check your dataset path and video files.")
else:
    r1_accuracy = (r1_correct / total_videos) * 100
    r3_accuracy = (r3_correct / total_videos) * 100
    r5_accuracy = (r5_correct / total_videos) * 100

    print("\n" + "="*30)
    print("      Evaluation Results")
    print("="*30)
    print(f"Total Videos Processed: {total_videos}")
    print(f"R@1 Accuracy: {r1_accuracy:.2f}% ({r1_correct}/{total_videos})")
    print(f"R@3 Accuracy: {r3_accuracy:.2f}% ({r3_correct}/{total_videos})")
    print(f"R@5 Accuracy: {r5_accuracy:.2f}% ({r5_correct}/{total_videos})")
    print("="*30)