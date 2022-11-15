# 模型集成

nvidia-smi

python ../own_code/ensemble.py \
--batch 512 --datetime predict --gpu 0 1 2 3 \
--data_file ../../raw_data/testB.json --label 36 \
--model_dict \
"[
    {'name': '../../user_data/models_after_train/2022_10_22_19_12_04/best_3.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '../../user_data/models_after_train/2022_10_27_07_38_29-8/best_3.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '../../user_data/models_after_train/2022_11_01_04_26_32/best_3.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '../../user_data/models_after_train/2022_11_03_19_41_25/best_1.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '../../user_data/models_after_train/2022_11_03_19_41_25/best_2.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '../../user_data/models_after_train/2022_11_05_05_55_17/best_0.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '../../user_data/models_after_train/2022_11_06_04_35_15/best_1.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '../../user_data/models_after_train/2022_11_06_04_35_15/best_2.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '../../user_data/models_after_train/2022_11_06_19_08_24/best_2.pt', 'model': 'model1', 'bert': '../own_code/pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
]" \
--model_weight "[1, 1, 1, 1, 1, 1, 1, 1, 1]" 