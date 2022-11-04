# 模型集成

python ensemble.py \
--batch 256 --datetime $(date +%Y_%m_%d_%H_%M_%S) --gpu 2 3 \
--data_file data/fewshot/testA.json --model_folder_path models --label 36 \
--model_dict \
"[
    {'name': '2022_10_28_15_41_45/best_1.pt', 'model': 'model5', 'bert': 'pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '2022_10_28_15_41_45/best_2.pt', 'model': 'model5', 'bert': 'pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
    {'name': '2022_10_28_15_41_45/best_3.pt', 'model': 'model5', 'bert': 'pretrained/nghuyong/ernie-3.0-base-zh',  'feature_layers': 4, 'dropout': 0.3, 'language': 'zh', 'swa': False},
]" \
--model_weight "[1, 1, 1]" \
# --score --single