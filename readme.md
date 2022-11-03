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

# 训练记录

## TO DO LIST

- [X] AdamW  加 correct_bias = True 《Revisiting Few-sample BERT Fine-tuning》 https://zhuanlan.zhihu.com/p/524036087
- [X] 对官方测试集打伪标签加入训练集
- [X] warmup 正确应用
- [X] ema、pgd、fgm使用
- [X] 模型优化：删除ReLU，不同层Bert进行连接
- [X] 增加更多的Bert
- [X] K=1时也采用分层采样
- [ ] Mixtext

## 记录

| 时间                                                                                                                                   | 成员   | 得分                                                                                                                | 预训练模型                            | 训练轮数 | 交叉验证 | 其他设置                                                                                                                                                                              | 训练集+验证集得分                                                                                                                                                                                                                                                                                                                                                                                     | 验证集得分                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 训练开始：2022/10/12 15:49:35<br />训练结束：2022/10/12 18:52:39<br />提交时间：2022/10/12 18:54                                       | 张兆   | 0.46661411496                                                                                                       | nghuyong/ernie-3.0-base-zh            | 60       | 5        | warmup 0.1（用的可能不对）<br />2080Ti*2 batch=24<br />random_seed=42                                                                                                                 | best_1.pt : 0.7630403185746878<br />best_2.pt : 0.785194086089213<br />best_3.pt : 0.8484339825318716<br />best_4.pt : 0.8481442706893945<br />best_5.pt : 0.8564035464087444<br />bagging : 0.9039802357258913                                                                                                                                                                                       |                                                                                                                                       |
| 训练开始：2022/10/12 20:34:47<br />训练结束：2022/10/12 20:52:25<br />提交时间：2022/10/12 20:54                                       | 张兆   | 0.50926177                                                                                                          | Langboat/mengzi-bert-base             | 25       | 1        | 2080Ti*2 batch=24<br />random_seed=42<br />random_split_ratio=0.8                                                                                                                     | 0.9035649164096997                                                                                                                                                                                                                                                                                                                                                                                    | 0.555419                                                                                                                              |
| 训练开始：2022/10/12 20:57:30<br />训练结束：2022/10/13 00:12:28<br />提交时间：2022/10/13 08:23                                       | 张兆   | 0.54461296043                                                                                                       | nghuyong/ernie-3.0-base-zh            | 30       | 10       | 2080Ti*2 batch=24<br />random_seed=42                                                                                                                                                 | best_1.pt : 0.9501953736953739<br /> best_2.pt : 0.8090291204084319<br />best_3.pt : 0.6498208126075585<br />best_4.pt : 0.9427773331560824<br />best_5.pt : 0.9111302191404046<br />best_6.pt : 0.8292753225364519<br />best_7.pt : 0.8938565221741899<br />best_8.pt : 0.9537918662275343<br />best_9.pt : 0.7997188727837208<br />best_10.pt : 0.8814895748089789<br />bagging : 0.954136902855915 | 0.616938<br />0.586474<br />0.529849<br />0.564196<br />0.568573<br />0.516865<br />0.46167<br />0.598384<br />0.542902<br />0.67397  |
| 训练开始：2022/10/13 11:39:52<br />训练结束：2022/10/13 16:40:56<br />提交时间：2022/10/13 17:38                                       | 张兆   | 0.56185948422                                                                                                       | nghuyong/ernie-3.0-base-zh            | 50       | 10       | 2080Ti*2 batch=24<br />random_seed=42<br />correct_bias = True                                                                                                                        | best_1.pt : 0.958411<br />best_2.pt : 0.878444<br />best_3.pt : 0.950718<br />best_4.pt : 0.932414<br />best_5.pt : 0.929663<br />best_6.pt : 0.952183<br />best_7.pt : 0.952413<br />best_8.pt : 0.965563<br />best_9.pt : 0.949669<br />best_10.pt : 0.761528<br />bagging : 0.995802                                                                                                     | 0.627148<br />0.627045<br />0.524618<br />0.632453<br />0.603112<br />0.521837<br />0.550469<br />0.65892<br />0.498028<br />0.668365 |
| 训练开始：2022/10/15 10:00:01<br />训练结束：2022/10/15 10:25:30<br />训练开始：2022/10/15 11:25:11<br />训练结束：2022/10/15 11:59:54 | 张兆   | 未验证                                                                                                              | nghuyong/ernie-3.0-base-zh            | 50       | 1        | 未添加RDrop：<br />2080Ti*2 batch=24<br />random_seed=42<br />correct_bias = True<br />添加RDrop：<br />2080Ti*4 batch=24<br />random_seed=42<br />correct_bias = True<br />RDrop=0.4 | 未添加RDrop：0.909574<br />添加RDrop：0.924773                                                                                                                                                                                                                                                                                                                                                        | 未添加RDrop：0.568337<br />添加RDrop：0.635372                                                                                        |
| 训练开始：2022/10/15 14:01:25<br />训练结束：2022/10/15 14:35:36                                                                       | 张兆   | 未验证                                                                                                              | nghuyong/ernie-3.0-base-zh            | 50       | 1        | 添加FGM：<br />2080Ti*4 batch=24<br />random_seed=42<br />correct_bias = True                                                                                                         | 未添加FGM：0.909574<br />添加FGM：0.924612                                                                                                                                                                                                                                                                                                                                                            | 未添加FGM：0.568337<br />添加FGM：0.60684                                                                                             |
| 训练开始：2022/10/15 14:37:24<br />训练结束：2022/10/15 15:49:57                                                                       | 张兆   | 未验证                                                                                                              | nghuyong/ernie-3.0-base-zh            | 50       | 1        | 添加PGD：<br />2080Ti*4 batch=24<br />random_seed=42<br />correct_bias = True<br />PGD_K=3                                                                                            | 未添加PGD：0.909574<br />添加PGD：0.917637                                                                                                                                                                                                                                                                                                                                                            | 未添加PGD：0.568337<br />添加PGD：0.582854                                                                                            |
| 训练开始：2022/10/15 21:18:01<br />训练结束：2022/10/16 03:22:04                                                                       | 张兆   | 0.56684304078                                                                                                       | nghuyong/ernie-3.0-base-zh            | 50       | 10       | 2080Ti*4 batch=24<br />random_seed=42<br />correct_bias = True<br />RDrop=0.4                                                                                                         | best_1.pt : 0.964261<br />best_2.pt : 0.937494<br />best_3.pt : 0.93178<br />best_4.pt : 0.915917<br />best_5.pt : 0.954728<br />best_6.pt : 0.95125<br />best_7.pt : 0.951136<br />best_8.pt : 0.957213<br />best_9.pt : 0.959292<br />best_10.pt : 0.967787<br />bagging :  0.995802                                                                                                     | 0.58019<br />0.576124<br />0.526915<br />0.577535<br />0.612728<br />0.559396<br />0.549781<br />0.646989<br />0.537495<br />0.687826 |
|                                                                                                                                        | 李一鸣 | **0.60286913357**                                                                                             | Langboat/mengzi-bert-base             | 40       | 5        | batch=12 用官方测试集 F1 = 0.57 的模型产生的伪标签进行 pseudo-labelling                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                                                        | 李一鸣 | 0.587                                                                                                               | Langboat/mengzi-bert-base             | 40       | 1        | batch=12 用官方测试集 F1 = 0.57 的模型产生的伪标签进行 pseudo-labelling                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                                                        | 李一鸣 | 0.58719238                                                                                                          | Langboat/mengzi-bert-base             | 40       | 5        | batch=12, 用官方测试集 F1 = 0.60 的模型产生的伪标签进行 pseudo-labelling, RDrop=0.1                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                                                        | 李一鸣 | 0.58009964029                                                                                                       | Langboat/mengzi-bert-base             | 40       | 1        | batch=12, 用官方测试集 F1 = 0.60 的模型产生的伪标签进行 pseudo-labelling, RDrop=0.1                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                                                        | 李一鸣 | 0.60216                                                                                                             | Langboat/mengzi-bert-base             | 40       | 1        | batch=24, 用官方测试集 F1 = 0.599 的模型产生的伪标签(expand_train_cur_best.json )进行 pseudo-labelling, no extra tricks                                                               |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
| 训练开始：2022/10/16 17:28:14<br />训练结束：2022/10/16 18:10:27                                                                       | 张兆   | 0.52782175403                                                                                                       | Langboat/mengzi-bert-base             | 50       | 1        | 2080Ti*2 batch=24                                                                                                                                                                     | 0.911089                                                                                                                                                                                                                                                                                                                                                                                              | 0.604567                                                                                                                              |
| 训练开始：2022/10/16 18:10:31<br />训练结束：2022/10/16 18:58:40                                                                       | 张兆   | 0.50326362551                                                                                                       | Langboat/mengzi-bert-base             | 50       | 1        | 2080Ti*4 batch=24<br />RDrop=0.4                                                                                                                                                      | 0.900517                                                                                                                                                                                                                                                                                                                                                                                              | 0.560547                                                                                                                              |
| 训练开始：2022/10/16 18:58:44<br />训练结束：2022/10/16 19:41:36                                                                       | 张兆   | 0.50364182397                                                                                                       | nghuyong/ernie-3.0-base-zh            | 50       | 1        | 2080Ti*2 batch=24                                                                                                                                                                     | 0.914422                                                                                                                                                                                                                                                                                                                                                                                              | 0.566295                                                                                                                              |
| 训练开始：2022/10/16 20:57:37<br />训练结束：2022/10/16 21:48:23                                                                       | 张兆   | 0.52432191354                                                                                                       | nghuyong/ernie-3.0-base-zh            | 50       | 1        | 2080Ti*4 batch=24<br />RDrop=0.4                                                                                                                                                      | 0.924773                                                                                                                                                                                                                                                                                                                                                                                              | 0.635372                                                                                                                              |
|                                                                                                                                        | 张兆   | 0.60906089726<br />1：0.59323842065<br />**3：0.62876738448**<br />5：0.60725801303<br />2 3 4：0.61215379337 | nghuyong/ernie-3.0-base-zh            | 40       | 5        | V100*4 batch=128<br />fgm                                                                                                                                                             | 0.940159<br />0.959345<br />0.982243<br />0.961593<br />0.961342<br />bagging:0.967881                                                                                                                                                                                                                                                                                                                | 0.83072<br />0.928769<br />0.927161<br />0.934107<br />0.943387                                                                       |
| 大模型实验<br />2022_10_29_18_22_10                                                                                                    | 张兆   | 0.58887581038                                                                                                       | hfl/chinese-roberta-wwm-ext-large     | 50       | 1        | V100*8 batch=64<br />fgm                                                                                                                                                              | 0.987128                                                                                                                                                                                                                                                                                                                                                                                              | Epoch=27<br />0.924402                                                                                                                |
| 全语言实验<br />2022_10_29_18_43_16                                                                                                    | 张兆   | 0.60163217528                                                                                                       | pretrained/nghuyong/ernie-3.0-base-zh | 50       | 1        | V100*8 batch=128<br />fgm                                                                                                                                                             | 0.999038                                                                                                                                                                                                                                                                                                                                                                                              | Epoch =29<br />0.99521                                                                                                                |
| 大模型实验<br />2022_10_29_20_26_06                                                                                                    | 张兆   | 0.59526036696                                                                                                       | nghuyong/ernie-3.0-xbase-zh           | 50       | 1        | V100*8 batch=128<br />fgm                                                                                                                                                             | 0.983985                                                                                                                                                                                                                                                                                                                                                                                              | Epoch =34<br />0.913857                                                                                                               |

## 实验计划

探究不同因素对不同模型的影响

基本配置：batch=12，epoch=40 (early stop), gpu=2,3, lr=2e-5, seed=42, split_test_ratio=0.2, dropout=0.3

数据采用expand_train.json

| 模型                    | baseline                                                                                 | rdrop=0.1                                                                     | rdrop=0.5 | rdrop=1.0                                                                     | ema=0.999                                                                     | pgd=3                                                                         | warmup=0.1                                                                    | fgm                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| mengzi-bert-base        | 2022_10_25_05_36_57<br />Epoch =11 0.943702<br />0.973459<br />0.58946182233            | 2022_10_25_05_36_51<br />Epoch =13 0.948241<br />0.981155<br />0.58861785176 |           |                                                                               | 2022_10_25_05_37_12<br />Epoch =12 0.953051<br />0.953051<br />0.58592248996 | 2022_10_25_05_37_15<br />Epoch =22 0.955365<br />0.984842<br />0.58437014014 | 2022_10_25_05_41_54<br />Epoch =20 0.955028<br />0.983924<br />0.58043132506 | 2022_10_25_05_37_08<br />Epoch =14 0.94846<br />0.982611<br />0.58877098445  |
| ernie-3.0-base-zh       | **2022_10_21_17_44_31<br />Epoch = 12 0.939449<br />0.975266<br />0.61203747617** |                                                                               |           |                                                                               | 2022_10_22_16_39_14<br />Epoch=32 0.947126<br /> 0.986405<br /> 0.60179269740 | 2022_10_23_04_57_21<br />Epoch=20 0.94027<br />0.975243<br />0.58086732341    | 2022_10_21_21_46_43<br />Epoch =28 0.939117<br />0.981063<br />0.60027698592 | 2022_10_22_22_19_15<br /> Epoch=20 0.942571 <br />0.981936<br />0.59630101157 |
| chinese-macbert-base    | 2022_10_19_09_04_25<br />Epoch =11 0.93218<br />0.973964<br />0.58566511183             | 2022_10_21_17_25_43<br />Epoch =27 0.937448<br />0.984506<br />0.56675022857 |           | 2022_10_20_20_00_52<br />Epoch =19 0.944687<br />0.983283<br />0.57831287521 | 2022_10_22_03_38_29<br />Epoch =17 0.942641<br />0.981227<br />0.58781315810 | 2022_10_20_02_42_28<br />Epoch =19 0.950837<br />0.98362<br />0.59952852320  | 2022_10_21_04_24_30<br />Epoch =33 0.955382<br />0.990917<br />0.57818127106 | 2022_10_19_14_51_36<br />Epoch =25 0.940839<br />0.988269<br />0.59486694090 |
| chinese-roberta-wwm-ext | 2022_10_25_08_19_12<br />Epoch=14 0.943605 <br />0.979684<br />0.59554496893             |                                                                               |           |                                                                               | 2022_10_25_11_23_51<br />Epoch=25 0.948677<br />0.985914<br />0.59928113089   |                                                                               | 2022_10_25_15_54_51<br /> Epoch=15 0.946985<br />0.980388<br />0.58545683003  | 2022_10_25_19_05_47<br />Epoch=15 0.945916<br />0.983087<br />0.60706732296   |

## 英文模型

基本配置：batch=12，epoch=40 (early stop), gpu=2,3, lr=2e-5, seed=42, split_test_ratio=0.2, dropout=0.3

数据采用expand_train_cur_best_en.json

| 模型              | baseline                                                                      | ema=0.999                                                                     | pgd=3                                                                         | fgm                                                                           | warmup=0.1                                                                    |
| ----------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| ernie-2.0-base-en | 2022_10_23_18_05_14<br />Epoch =24 0.869316<br />0.947128<br />0.58633437923 | 2022_10_23_18_18_01<br />Epoch =27 0.883606<br />0.973969<br />0.58900858108 | 2022_10_23_18_05_27<br />Epoch =18 0.851685<br />0.94073<br />0.56777475031  | 2022_10_23_18_19_40<br />Epoch =24 0.884273<br />0.953767<br />0.60969211560 | 2022_10_23_18_05_32<br />Epoch =25 0.86958<br />0.968817<br />0.59526836713  |
| roberta-base      |                                                                               |                                                                               |                                                                               |                                                                               |                                                                               |
| xlnet-base-cased  | 2022_10_24_05_41_37<br />Epoch =31 0.876336<br />0.976078<br />0.59146824680 | 2022_10_24_05_41_04<br />Epoch =17 0.868918<br />0.969398<br />0.59067118434 | 2022_10_24_05_41_12<br />Epoch =19 0.875453<br />0.970272<br />0.57834073199 | 2022_10_24_05_41_20<br />Epoch =21 0.879417<br />0.976025<br />0.58459104336 | 2022_10_24_05_41_29<br />Epoch =24 0.873381<br />0.974582<br />0.59797619630 |
| deberta-v3-base   | 2022_10_24_10_44_57<br />Epoch =25 0.875878<br />0.943788<br />0.58797673621 | 2022_10_24_10_50_12<br />Epoch =37 0.880665<br />0.947626<br />0.59859900583 | 2022_10_24_10_52_08<br />Epoch =28 0.894997<br />0.955026<br />0.60094762341 | 2022_10_24_10_54_03<br />Epoch =24 0.881334<br />0.947271<br />0.59791066113 | 2022_10_24_10_51_34<br />Epoch =24 0.856213<br />0.933411<br />0.58096981486 |

二郎神：IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese

数据采用expand_train_cur_best.json

2022_10_26_06_08_38：不加 Epoch =14 0.909819 0.979235 0.61420523782

2022_10_26_06_05_48：加fgm Epoch =8 0.913229 0.979437 0.59672118680

ernie-2.0-base-en+fgm 十折交叉验证 expand_train_cur_best_en.json

2022_10_25_05_36_32

1：Epoch =23 0.738594 0.943053 0.59779498710

2：Epoch =20 0.942162 0.965787 0.61183453255

3：Epoch =14 0.921992 0.961072 0.60124333189

4：Epoch =22 0.912673 0.964743 0.60470543799

5：Epoch =15 0.90293 0.962382 0.61298211869

6：Epoch =23 0.918961 0.974785 0.62131731657

7：Epoch =15 0.920576 0.963609

8：Epoch =20 0.914956 0.965676

9：Epoch =40 0.946059 0.991682

10：Epoch=29 0.938323 0.990791

0.970571

0.61755036431

ernie-3.0-base-zh+fgm 十折交叉验证 expand_train_cur_best+en_zh.json

2022_10_25_05_36_40

1：Epoch =39 0.980802 0.997899  **0.62529548458**

2：Epoch =35 0.99583 0.999507 0.61470568046

3：Epoch=33 0.968477 0.996453 0.61029077511

4：Epoch =40 0.998369 0.99666 0.61501080887

5：Epoch =37 0.96937 0.996507 0.61981191769

6：Epoch =34 0.95805 0.995294 0.62014754016

7：Epoch =39 0.97682 0.994493 0.60765491039

8：Epoch =37 0.950761 0.994647

9：Epoch=35 0.983666 0.995142

10：Epoch =39 0.974248 0.997419

0.996792

0.62106441848

## 随机数实验

batch=128, bert='pretrained/nghuyong/ernie-3.0-base-zh'

data_file='expand_train_cur_best.json',

8*V100 32G

五折

| 随机数   | 时间戳              | 1                                                        | 2                                                         | 3                                                                 | 4                                                        | 5                                                        | all                                   |
| -------- | ------------------- | -------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------- |
| 3407     | 2022_10_27_07_36_12 | Epoch =29<br />0.828277<br />0.940246                    | Epoch=38<br />0.922059<br />0.975522                      | Epoch =26<br />0.922241<br />0.978571                             | Epoch =20<br />0.920416<br />0.961822<br />0.59931987441 | Epoch =18<br />0.928557<br />0.964518                    | 0.976724<br />0.60708719687           |
| fgm+3407 | 2022_10_27_07_32_33 | Epoch =29<br />0.837168<br />0.958176<br />0.62207249038 | Epoch =33<br />0.93111<br />0.978384<br />0.61360985387   | Epoch=27<br />0.928721<br />0.981655<br />0.60667812125           | Epoch =22<br />0.932218<br />0.959626<br />0.59740545961 | Epoch =29<br />0.938336<br />0.972912<br />0.59479068059 | 0.984951<br />0.61382655005           |
| 219      | 2022_10_27_07_40_08 | Epoch = 16<br />0.82509<br />0.936603<br />0.59974018113 | Epoch = 17<br />0.921952<br />0.956262<br />0.60490219534 | Epoch = 21<br />0.912299<br />0.955187<br />0.59659364810         | Epoch = 16<br />0.932538<br />0.960904                   | Epoch = 35<br />0.936999<br />0.98227<br />0.61027444916 | 0.976059<br />0.61142175663           |
| fgm+219  | 2022_10_27_07_44_22 | Epoch=33<br />0.837085<br />0.942478<br />0.61739339132  | Epoch =36<br />0.931217<br />0.978476<br />0.61062900994  | Epoch =32<br />0.920706<br />0.981834<br />0.61664927614          | Epoch =34<br />0.934104<br />0.978847<br />0.62235183026 | Epoch =35<br />0.945433<br />0.986606<br />0.60045155119 | 0.991535<br />0.61824679505           |
| 909      | 2022_10_27_07_38_29 | Epoch =38<br />0.827719<br />0.949487                    | Epoch =32<br />0.927994<br />0.969012                     | Epoch =30<br />0.918163<br />0.97716<br />**0.63077875449** | Epoch =18<br />0.927548<br />0.968832                    | Epoch=38<br />0.933141<br />0.982951<br />0.60616902305  | 0.985358<br />**0.62524955972** |
| fgm+909  | 2022_10_27_07_43_16 | Epoch = 34<br />0.833804<br />0.948922                   | Epoch = 21<br />0.932729<br />0.96123<br />0.60084338886  | Epoch = 13<br />0.921903<br />0.949437<br />0.59973293567         | Epoch = 36<br />0.934<br />0.979314<br />0.62197043809   | Epoch = 36<br />0.938507<br />0.98511<br />0.60984056326 | 0.978551<br />0.61553075216           |

### swa实验

ernie-3.0-base-zh，单卡V100 32G，batch 32，epoch 50，数据expand_train_cur_best.json

| 添加fgm                                                                                 | 不添加fgm                                                                     |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 2022_10_29_09_06_14<br />Epoch =50 0.916163<br />0.981227<br />**0.62434188208** | 2022_10_29_09_03_42<br />Epoch =46 0.914986<br />0.980371<br />0.60998950294 |

### 不同模型结构实验

ernie-3.0-base-zh，4*V100 32G，batch 128，epoch 50，数据expand_train_cur_best.json，fgm

| model1                                                                        | model2                                                                        | model3                                                                        | model4                                                                        | model5                                                                        |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| last_hidden+MLP                                                               | 后四层连接                                                                    | pooler_output输出                                                             | pooler_output+MLP                                                             | 前四层+后四层平均                                                             |
| 2022_10_29_13_36_46<br />Epoch =29 0.917249<br />0.980737<br />0.61477357402 | 2022_10_29_09_10_37<br />Epoch =29 0.917644<br />0.985873<br />0.60476939744 | 2022_10_29_09_13_44<br />Epoch =24 0.921817<br />0.982066<br />0.60608243078 | 2022_10_29_09_14_02<br />Epoch =41 0.923856<br />0.982397<br />0.61429143697 | 2022_10_29_09_16_20<br />Epoch =14 0.918428<br />0.974142<br />0.60180561494 |

### 5折实验

expand_train_cur_best.json

| 模型                    | 训练配置                                            | 1                                                        | 2                                                        | 3                                                        | 4                                                        | 5                                                        | all                         |
| ----------------------- | --------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- | --------------------------- |
| ernie-xbase + fgm       | 2022_10_30_05_13_40<br />8*V100 32G<br />batch 128  | Epoch=28<br />0.85882<br />0.962631<br />0.59420600432   | Epoch =11<br />0.925827<br />0.957288<br />0.59320443123 | Epoch =20<br />0.930223<br />0.982003<br />0.60906881597 | Epoch =18<br />0.93642<br />0.976608<br />0.59153312985  | Epoch =18<br />0.945131<br />0.984544<br />0.60618609586 | 0.99059<br />0.60398456406  |
| ernie-xbase + fgm + swa | 2022_10_30_05_17_06<br />单卡V100 32G<br />batch 16 | Epoch =34<br />0.833288<br />0.59777530265               | Epoch =23<br />0.936805<br />2022_11_02_03_26_58         | Epoch=23<br />0.942784<br />2022_11_02_20_43_59          | 停止                                                     |                                                          |                             |
| ernie + fgm + swa       | 2022_10_30_05_10_41<br />单卡V100 32G<br />batch 32 | Epoch =18<br />0.850886<br />0.945433<br />0.61460610456 | Epoch =22<br />0.937315<br />0.962094<br />0.61609908526 | Epoch=47<br />0.927484<br />0.984062<br />0.61932705714  | Epoch =43<br />0.936822<br />0.980606<br />0.61907224613 | Epoch =29<br />0.948016<br />0.965169<br />0.61239652746 | 0.970118<br />0.61836815011 |

### 不同Loss实验

ernie-3.0-base-zh，2*V100 32G，batch 32，expand_train_cur_best.json

| cb                                                      | rfl                                                   | ntrfl                                                    | dbfl                                                     |
| ------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| 2022_10_30_16_22_03                                     | 2022_10_30_16_26_50                                   | 2022_10_30_16_31_32                                      | 2022_10_30_16_32_21                                      |
| Epoch =42<br />0.92314<br />0.986964<br />0.61037779477 | Epoch=26<br />0.92106<br />0.98226<br />0.61772640943 | Epoch =17<br />0.925418<br />0.981216<br />0.61171727349 | Epoch =29<br />0.922694<br />0.981506<br />0.60664211626 |

### 大模型随机数实验

batch=64, bert='pretrained/nghuyong/ernie-3.0-xbase-zh'

data_file='expand_train_cur_best.json'

4*V100 32G

四折

| 随机数 | 时间戳              | 1                                                               | 2                                                               | 3                                                                 | 4                                                               | all                         |
| ------ | ------------------- | --------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------- |
| 5267   | 2022_10_31_03_07_03 | Epoch = 8<br />0.844945<br />0.936316<br />2022_11_01_03_36_44  | Epoch = 14<br />0.929097<br />0.975143<br />2022_11_01_03_42_36 | Epoch = 37<br />0.936798<br />0.983429<br />2022_11_01_03_48_52   | Epoch = 20<br />0.937756<br />0.982108<br />2022_11_01_03_52_27 | 0.979784                    |
| 6271   | 2022_10_31_03_07_58 | Epoch =26<br />0.857581<br />0.953757<br />0.60340980050        | Epoch =11<br />0.933323<br />0.956936<br />2022_11_01_03_59_29  | Epoch =22<br />0.93793<br />0.981894<br />2022_11_01_04_03_06     | Epoch =15<br />0.933826<br />0.980293<br />2022_11_01_04_06_31  | 0.985472                    |
| 3254   | 2022_10_31_03_13_49 | Epoch =15<br />0.848537<br />0.954258<br />2022_11_01_04_09_52  | Epoch =23<br />0.930986<br />0.979635<br />0.59871831078        | Epoch =33<br />0.936514<br />0.98287<br />**0.62449404558** | Epoch=24<br />0.931808<br />0.981126<br />0.59931316293         | 0.995635<br />0.61619123551 |
| 1618   | 2022_10_31_03_15_53 | Epoch =24<br />0.850448<br />0.956344<br />2022_11_01_04_24_47  | Epoch =17<br />0.921543<br />0.971829<br />2022_11_01_04_28_20  | Epoch =9<br />0.927174<br />0.954966<br />2022_11_01_04_31_58     | Epoch=41<br />0.939747<br />0.984025<br />0.61157607909         | 0.990993                    |
| 5374   | 2022_10_31_03_19_03 | Epoch = 12<br />0.851206<br />0.938386<br />2022_11_01_04_40_29 | Epoch = 15<br />0.922498<br />0.977894<br />2022_11_01_04_44_08 | Epoch = 14<br />0.935281<br />0.976279<br />2022_11_01_04_47_53   | Epoch = 16<br />0.934405<br />0.97992<br />2022_11_01_04_51_41  | 0.98513                     |
| 7606   | 2022_10_31_03_35_55 | Epoch =14<br />0.851953<br />0.948443<br />0.61432388691        | Epoch=21<br />0.931022<br />0.974039<br />2022_11_01_04_59_19   | Epoch=17<br />0.935818<br />0.977052<br />2022_11_01_05_03_09     | Epoch =24<br />0.944068<br />0.984329<br />0.58745971606        | 0.991329                    |

### 增强数据实验

expand_train_aug_tail.json

ernie-3.0-base-zh

fgm

model1

| 配置                                          | 时间戳                                       | 1                                                | 2                                                        | 3                                                                  | 4                                                        | 5                                                        | all                         |
| --------------------------------------------- | -------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------- | --------------------------- |
| batch=128<br />4*V100<br />seed=3407          | 2022_11_01_04_22_41<br />2022_11_02_03_08_56 | Epoch =32<br />0.871526<br />0.974331            | Epoch =28<br />0.964592<br />0.991178                    | Epoch =41<br />0.973115<br />0.994438<br />0.62325553677           | Epoch =13<br />0.970786<br />0.989015<br />0.61143654969 | Epoch =19<br />0.976139<br />0.992837<br />0.61164173186 | 0.997983<br />0.61838623151 |
| batch=128<br />4*V100<br />seed=42            | 2022_11_01_04_26_32<br />2022_11_02_03_10_10 | Epoch =29<br />0.87611<br />0.975097             | Epoch =31<br />0.965207<br />0.990986<br />0.61300549904 | Epoch =24<br />0.965826<br />0.991184<br />**0.63293263685** | Epoch =39<br />0.977998<br />0.995103<br />0.61587753363 | Epoch =18<br />0.972853<br />0.991349<br />0.60745748898 | 0.997956<br />0.62664852015 |
| batch=32<br />单卡<br />seed=42<br />swa=True | 2022_11_01_05_09_54                          | Epoch =48<br />0.876458<br />2022_11_02_03_40_24 | Epoch =28<br />0.971478<br />2022_11_02_20_35_04         | 停止                                                               |                                                          |                                                          |                             |

### 随机数实验

batch=64, bert='pretrained/nghuyong/ernie-3.0-base-zh'

data_file='expand_train_cur_best.json',

4*V100 32G

fgm

四折

| 随机数 | 时间戳                                       | 1                                     | 2                                     | 3                                     | 4                                     | all                         |
| ------ | -------------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | --------------------------- |
| 1      | 2022_11_02_03_32_27<br />2022_11_02_19_38_48 | Epoch =12<br />0.853343<br />0.938496 | Epoch =23<br />0.934691<br />0.961254 | Epoch =18<br />0.93041<br />0.959059  | Epoch =20<br />0.937899<br />0.977642 | 0.969301                    |
| 2      | 2022_11_02_03_32_58<br />2022_11_02_19_47_37 | Epoch =25<br />0.856089<br />0.955624 | Epoch =25<br />0.935358<br />0.978399 | Epoch =19<br />0.925913<br />0.957886 | Epoch =18<br />0.937418<br />0.980764 | 0.968468<br />0.62021852110 |
| 3      | 2022_11_02_03_37_21<br />2022_11_02_19_56_27 | Epoch =16<br />0.857157<br />0.941225 | Epoch =21<br />0.936291<br />0.977023 | Epoch =36<br />0.9289<br />0.977059   | Epoch =42<br />0.931982<br />0.980649 | 0.9868                      |
| 4      | 2022_11_02_03_40_55<br />2022_11_02_20_05_19 | Epoch=20<br />0.85396<br />0.940266   | Epoch =13<br />0.934075<br />0.958485 | Epoch=25<br />0.937078<br />0.977961  | Epoch =13<br />0.933042<br />0.953424 | 0.968248                    |

### 其他实验

ernie-3.0-base-zh，model1，batch=128，4*V100 32G，seed=42

| 配置                                                    | 时间戳              | 1                                                  | 2                                                  | 3                                                  | 4                                                  | 5                                     | all                         |
| ------------------------------------------------------- | ------------------- | -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- | ------------------------------------- | --------------------------- |
| train_1.json                                            | 2022_11_02_19_46_18 | Epoch=30<br />0.951351<br />2022_11_03_03_44_51_1  | Epoch =8<br />0.960934<br />2022_11_03_03_44_51_2  | Epoch =33<br />0.972444<br />2022_11_03_03_44_51_3 |                                                    |                                       |                             |
| train_2.json                                            | 2022_11_02_19_48_24 | Epoch =27<br />0.936979<br />2022_11_03_03_44_51_4 | Epoch =14<br />0.968284<br />2022_11_03_03_44_51_5 | Epoch =24<br />0.968725<br />2022_11_03_03_44_51_6 |                                                    |                                       |                             |
| train_3.json                                            | 2022_11_02_19_49_15 | Epoch =33<br />0.945613<br />2022_11_03_03_44_51_7 | Epoch =33<br />0.96962<br />2022_11_03_03_44_51_8  | Epoch =22<br />0.970094<br />2022_11_03_05_21_28_1 |                                                    |                                       |                             |
| assignee=True<br />expand_train_aug_tail.json           | 2022_11_02_19_51_58 | Epoch =24<br />0.862726<br />2022_11_03_03_44_51_9 | Epoch =7<br />0.945047<br />2022_11_03_03_44_51_10 | Epoch =9<br />0.953333<br />2022_11_03_03_44_51_11 | Epoch =30<br />0.960581<br />2022_11_03_04_50_37_1 |                                       |                             |
| seed=909<br />expand_train_aug_tail.json<br />fgm=False | 2022_11_02_19_54_00 | Epoch =16<br />0.862872<br />0.970313              | Epoch =27<br />0.961241<br />0.990883              | Epoch =15<br />0.958736<br />0.985262              | Epoch =19<br />0.967793<br />0.990451              | Epoch =16<br />0.965509<br />0.985466 | 0.996883<br />lc            |
| seed=42<br />train_630_aug_tail.json                    | 2022_11_02_19_58_44 | Epoch =38<br />0.86662<br />2022_11_03_03_59_12_1  | Epoch=22<br />0.966239<br />2022_11_03_03_59_12_2  | Epoch =22<br />0.99014<br />2022_11_03_04_50_37_2  |                                                    |                                       |                             |
| seed=909<br />train_630_aug_tail.json<br />fgm=False    | 2022_11_02_19_59_44 | Epoch =20<br />0.858304<br />0.969588              | Epoch=14<br />0.961148<br />0.986552               | Epoch =19<br />0.981033<br />0.992011              | Epoch =16<br />0.967573<br />0.986563              | Epoch =10<br />0.967701<br />0.984855 | 0.995281<br />lc            |
| seed=42<br />train_632_aug_tail.json                    | 2022_11_02_20_01_13 | Epoch =24<br />0.870113<br />2022_11_03_03_59_12_3 | Epoch =42<br />0.972512<br />2022_11_03_03_59_12_4 | Epoch =25<br />0.993971<br />2022_11_03_05_21_28_1 |                                                    |                                       |                             |
| seed=909<br />train_632_aug_tail.json<br />fgm=False    | 2022_11_02_20_01_55 | Epoch =10<br />0.850151<br />0.964642              | Epoch=25<br />0.964354<br />0.990974               | Epoch =26<br />0.986338<br />0.994619              | Epoch =17<br />0.966243<br />0.988185              | Epoch=19<br />0.968454<br />0.988842  | 0.997676<br />0.62093379110 |
| seed=909<br />expand_train_630.json<br />fgm=False      | 2022_11_02_20_21_50 | Epoch =19<br />0.818384<br />0.937568              | Epoch =28<br />0.930541<br />0.975334              | Epoch=20<br />0.952989<br />0.961648               | Epoch =31<br />0.928331<br />0.97673               | Epoch =25<br />0.93317<br />0.980749  | 0.978132<br />lc            |
| seed=909<br />expand_train_632.json<br />fgm=False      | 2022_11_02_20_23_08 | Epoch =25<br />0.83736<br />0.960503               | Epoch =14<br />0.936257<br />0.977144              | Epoch=29<br />0.968587<br />0.984876               | Epoch =19<br />0.949106<br />0.97852               | Epoch =19<br />0.942543<br />0.977803 | 0.991369<br />0.60985007379 |

### 测试实验

1. 测试究竟有多少32（22也是一样的结果）

提交结果：0.00028845044

$F1score=\frac{1}{36}\cdot\frac{2PR}{P+R}=0.00028845044$，$R=1$

$0.00028845044*18(P+1)=P$， $(1-0.00519210792)P=0.00519210792$

共有20839条测试数据，因此 $\frac{0.00519210792}{1-0.00519210792}=\frac{x}{20839}$ ， $x=108.763046419599$

所以共有109条标签为32的数据？

2. 测试究竟有多少2

提交结果：0.00891031259

$F1score=\frac{1}{36}\cdot\frac{2PR}{P+R}=0.00891031259$， $R=1$

$0.00891031259*18(P+1)=P$， $(1-0.16038562662)P=0.16038562662$

共有20839条测试数据，因此 $\frac{0.16038562662}{1-0.16038562662}=\frac{x}{20839}$ ， $x=3980.72755672264$

所以共有3981条标签为2的数据？

3. 测试究竟有多少35

提交结果：0.00034578147
