import os
import random
import torch
import cv2
from loguru import logger

from demo.utils import (
    retrieve_text_with_saved_features,
    _frame_from_video,
    setup_internvideo2,
    save_text_features,
    get_text_feat_dict
)

class InternVideoZeroShotClassifier:
    def __init__(self, config, captions, save_path='text_feature_cache/classification_text_features.npy', use_prompt=False, prompt_templates=None):
        """
        初始化 InternVideoZeroShotClassifier

        Args:
            config: 配置对象，从 `Config.from_file()` 获得
            captions: 类别名称（原始或处理后的），如 ['brush_hair', 'ride_bike', ...]
            save_path: 文本特征缓存路径
            use_prompt: 是否使用 prompt 模板
            prompt_templates: 模板列表，如 kinetics_templates_action_clip
        """
        self.config = config
        self.replace_ = True
        self.save_path = save_path
        self.use_prompt = use_prompt
        self.token_captions = self._build_text_captions(captions, prompt_templates)
        self.class_names = captions  # 原始类别名（如 "brush_hair"）
        logger.debug(self.token_captions)
        
        logger.info("Setting up InternVideo2 model...")
        self.model, self.tokenizer = setup_internvideo2(config)
        logger.info("Model loaded.")

        # 计算或加载文本特征
        # os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        # if not os.path.exists(self.save_path):
        #     logger.info("Computing and saving text features...")
        #     save_text_features(self.token_captions, self.model, save_path=self.save_path)
        # else:
        #     logger.info("Loading cached text features...")
        # self.text_features = torch.load(self.save_path, map_location='cpu')

        
        self.model.to(torch.device('cuda'))
        
        # 计算文本特征
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            text_feat_d = get_text_feat_dict(self.token_captions, self.model, {})
        
            # 构建保存数据结构
            self.text_features = {
                'texts': self.token_captions,
                'features': [text_feat_d[t].cpu() for t in self.token_captions]
            }

    def _build_text_captions(self, captions, templates):
        if not self.use_prompt:
            if self.replace_:
                return [cap.replace('_', ' ') for cap in captions]
            else:
                return captions
        assert templates is not None and isinstance(templates, list), "Prompt templates must be provided if use_prompt=True"
        if self.replace_:
            return [random.choice(templates).format(cap.replace('_', ' ')) for cap in captions]
        else:
            return [random.choice(templates).format(cap) for cap in captions]

    def classify_video(self, video_path, topk=5):
        """
        对单个视频进行零样本分类

        Args:
            video_path: 视频文件路径
            topk: 返回前 topk 个预测标签

        Returns:
            (topk_texts, probs): 返回预测文本标签及其概率（均为列表）
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise IOError(f"Failed to open video: {video_path}")

        frames = [x for x in _frame_from_video(video)]
        video.release()

        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")

        texts, probs = retrieve_text_with_saved_features(
            frames,
            self.text_features,
            model=self.model,
            topk=topk,
            config=self.config
        )
        return texts, probs

    def evaluate_directory(self, dataset_root, topk=(1, 3, 5)):
        """
        遍历整个数据集并评估 R@1, R@3, R@5

        Args:
            dataset_root: 数据集根目录，格式为 DATASET_ROOT/class_name/video_file
            topk: Tuple[int]，默认计算 R@1/3/5

        Returns:
            dict: {'r1': float, 'r3': float, 'r5': float}
        """
        from tqdm import tqdm

        r1_correct, r3_correct, r5_correct = 0, 0, 0
        total = 0

        for idx, class_name in enumerate(tqdm(self.class_names, desc="Evaluating Classes")):
            # logger.debug(class_name)
            # logger.debug(self.token_captions[idx])
            class_dir = os.path.join(dataset_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            for file in os.listdir(class_dir):
                if not file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                video_path = os.path.join(class_dir, file)
                try:
                    texts, _ = self.classify_video(video_path, topk=max(topk))
                    total += 1
                    # logger.debug(texts[:1])
                    if self.token_captions[idx] in texts[:1]:
                        r1_correct += 1
                    if self.token_captions[idx] in texts[:3]:
                        r3_correct += 1
                    if self.token_captions[idx] in texts[:5]:
                        r5_correct += 1
                except Exception as e:
                    logger.warning(f"Failed to process {video_path}: {e}")

        results = {
            'total': total,
            'r1': r1_correct / total * 100 if total else 0,
            'r3': r3_correct / total * 100 if total else 0,
            'r5': r5_correct / total * 100 if total else 0
        }
        logger.info(f"Evaluation done. R@1: {results['r1']:.2f}%, R@3: {results['r3']:.2f}%, R@5: {results['r5']:.2f}%")
        return results
