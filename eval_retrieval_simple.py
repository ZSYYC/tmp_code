import os
from demo_config import Config, eval_dict_leaf
from dataset.text_prompt import kinetics_templates_action_clip
from InternVideoZeroShotClassifier import InternVideoZeroShotClassifier
# 1. 加载配置
config = Config.from_file('/root/InternVideo/InternVideo2/multi_modality/scripts/evaluation/clip/zero_shot/S14/config_hmdb51.py')
config = eval_dict_leaf(config)

# 2. 构造类别列表
DATASET_ROOT = '/root/hmdb51'
class_names = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]

# 3. 初始化分类器
classifier = InternVideoZeroShotClassifier(
    config=config,
    captions=class_names,
    save_path='text_feature_cache/classification_text_features.npy',
    use_prompt=True,
    prompt_templates=kinetics_templates_action_clip
)

# 4. 评估全数据集
results = classifier.evaluate_directory(DATASET_ROOT)
print(results)

# 或对单个视频分类
texts, probs = classifier.classify_video("/root/hmdb51/catch/Ball_hochwerfen_-_Rolle_-_Ball_fangen_(Timo_3)_catch_f_cm_np1_le_goo_0.avi")
print("Predictions:", texts[0], probs[0])
