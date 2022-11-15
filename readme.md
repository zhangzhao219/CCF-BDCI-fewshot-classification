# CCF-BDCI 小样本数据分类任务

## 参数设置

```python
parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--predict', action='store_true', help='Whether to predict')
parser.add_argument('--predict_with_score', action='store_true', default=False, help='Whether to predict and output score')

parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--epoch', type=int, default=50, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')
parser.add_argument('--early_stop',type=int, default=10, help='Early Stop Epoch')

parser.add_argument('--data_folder_dir', type=str, required=True, help='Data Folder Location')
parser.add_argument('--data_file', type=str, help='Data Filename')
parser.add_argument('--label', type=int, default=36, help='label num')

parser.add_argument('--checkpoint', type=int, default=0, help='Use checkpoint')
parser.add_argument('--load', action='store_true', help='load from checkpoint')
parser.add_argument('--load_pt', type=str, help='load from checkpoint')
parser.add_argument('--save', action='store_true', help='Whether to save model')

parser.add_argument('--bert', type=str, required=True, help='Choose Bert')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout ratio')
parser.add_argument('--feature_layer', type=int, default=4, help='feature layers num')
parser.add_argument('--freeze', type=int, default=0, help='freeze bert parameters')
parser.add_argument('--model', type=str, required=True, help='Model type')

parser.add_argument('--en', action='store_true', help='whether to use English model')

parser.add_argument('--K', type=int, default=1, help='K-fold')
parser.add_argument('--split_test_ratio', type=float, default=0.2, help='if no Kfold, split test ratio')

parser.add_argument('--awp', type=int, default=-1, help='AWP attack start epoch')
parser.add_argument('--ema', type=float, default=0.0, help='EMA decay')
parser.add_argument('--fgm', action='store_true', help='FGM attack')
parser.add_argument('--fl',  action='store_true', default=False, help='Whether to use focal loss combined with ce loss')
parser.add_argument('--mixif', action='store_true', help='mixup training')
parser.add_argument('--pgd', type=int, default=0, help='PGD K')
parser.add_argument('--rdrop', type=float, default=0.0, help='RDrop kl_weight')
parser.add_argument('--sce',  action='store_true', help='Whether to use symmetric cross entropy loss')
parser.add_argument('--swa', action='store_true', help='swa ensemble')
parser.add_argument('--warmup', type=float, default=0.0, help='warm up ratio')

args = parser.parse_args()
```

## 训练

```bash
python main.py \
--train \
--batch 12 --board --datetime ${TIMESTAMP} --epoch 50 --gpu ${GPU} --lr 2e-5 --seed ${SEED} --early_stop 10 \
--data_folder_dir fewshot --data_file ${TRAIN_FILE} --label ${LABEL} \
--checkpoint 20 --save \
--bert ${BERT} --dropout ${DROPOUT} --feature_layer 4 --freeze 0 --model ${MODEL} \
--K ${K} --split_test_ratio 0.2 --swa
# --en \
# --awp 1 --ema 0.999 --fgm --fl --mixif --pgd 3 --rdrop 0.1 --sce --swa --warmup 0.1
```

## 测试

```bash
python main.py \
--test \
--batch ${TEST_BATCH} --datetime ${TIMESTAMP} --gpu ${GPU} \
--data_folder_dir fewshot --data_file ${TRAIN_FILE} --label ${LABEL} \
--bert ${BERT} --dropout ${DROPOUT} --feature_layer 4 --freeze 0 --model ${MODEL} \
--K ${K} --swa
# --en \
```

## 推理

```bash
python main.py \
--predict \
--batch ${TEST_BATCH}  --datetime ${TIMESTAMP} --gpu ${GPU} \
--data_folder_dir fewshot --data_file testA.json --label ${LABEL} \
--bert ${BERT} --dropout ${DROPOUT} --feature_layer 4 --freeze 0 --model ${MODEL} \
--K ${K} --swa
# predict_with_score \
# --en \
```

## 打包

```bash
python pack.py --datetime 2022_10_13_10_59_28 --score 0.0001245
```

## 伪标签

```bash
python add_pseudo_labels.py \
--predict_csv result.csv \
--corpus_json testA.json --origin_train_json train.json \
--data_folder_dir fewshot
```

## 模型集成

```bash
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
```
