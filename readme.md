# CCF-BDCI 小样本数据分类任务

## 参数设置

```python
parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--predict', action='store_true', help='Whether to predict')
parser.add_argument('--predict_with_score', action='store_true', default=False, help='Whether to predict')

parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--epoch', type=int, default=50, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')

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
parser.add_argument('--freeze', type=int, default=8, help='freeze bert parameters')

parser.add_argument('--ema', type=float, default=0.0, help='EMA decay')
parser.add_argument('--fgm', action='store_true', help='FGM attack')
parser.add_argument('--K', type=int, default=1, help='K-fold')
parser.add_argument('--pgd', type=int, default=0, help='PGD K')
parser.add_argument('--rdrop', type=float, default=0.0, help='RDrop kl_weight')
parser.add_argument('--split_test_ratio', type=float, default=0.2, help='if no Kfold, split test ratio')
parser.add_argument('--warmup', type=float, default=0.0, help='warm up ratio')

args = parser.parse_args()
```

## 训练

```bash
python main.py \
--train \
--batch 24 --board --datetime ${TIMESTAMP} --epoch 50 --gpu 2 3 --lr 2e-5 --seed ${SEED} \
--data_folder_dir fewshot --data_file train.json --label 36 \
--checkpoint 25 --save \
--bert ${BERT} --dropout 0.4 --feature_layer 4 --freeze 8 \
--K ${K} --split_test_ratio 0.2
# --ema 0.999 --fgm --pgd 3 --rdrop 0.4 --warmup 0.1
```

## 测试

```bash
python main.py \
--test \
--batch 512 --datetime ${TIMESTAMP} --gpu 2 3 --seed ${SEED} \
--data_folder_dir fewshot --data_file train.json --label 36 \
--bert ${BERT} --dropout 0.4 --feature_layer 4 \
--K ${K}
```

## 推理

```bash
python main.py \
--predict \
--batch 512  --datetime ${TIMESTAMP} --gpu 2 3 --seed ${SEED} \
--data_folder_dir fewshot --data_file testA.json --label 36 \
--bert ${BERT} --dropout 0.4 --feature_layer 4 \
--K ${K}
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

| 时间                                                                                                                                   | 成员   | 得分                                                                                                                | 预训练模型                 | 训练轮数 | 交叉验证 | 其他设置                                                                                                                                                                              | 训练集+验证集得分                                                                                                                                                                                                                                                                                                                                                                                     | 验证集得分                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------- | -------------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 训练开始：2022/10/12 15:49:35<br />训练结束：2022/10/12 18:52:39<br />提交时间：2022/10/12 18:54                                       | 张兆   | 0.46661411496                                                                                                       | nghuyong/ernie-3.0-base-zh | 60       | 5        | warmup 0.1（用的可能不对）<br />2080Ti*2 batch=24<br />random_seed=42                                                                                                                 | best_1.pt : 0.7630403185746878<br />best_2.pt : 0.785194086089213<br />best_3.pt : 0.8484339825318716<br />best_4.pt : 0.8481442706893945<br />best_5.pt : 0.8564035464087444<br />bagging : 0.9039802357258913                                                                                                                                                                                       |                                                                                                                                       |
| 训练开始：2022/10/12 20:34:47<br />训练结束：2022/10/12 20:52:25<br />提交时间：2022/10/12 20:54                                       | 张兆   | 0.50926177                                                                                                          | Langboat/mengzi-bert-base  | 25       | 1        | 2080Ti*2 batch=24<br />random_seed=42<br />random_split_ratio=0.8                                                                                                                     | 0.9035649164096997                                                                                                                                                                                                                                                                                                                                                                                    | 0.555419                                                                                                                              |
| 训练开始：2022/10/12 20:57:30<br />训练结束：2022/10/13 00:12:28<br />提交时间：2022/10/13 08:23                                       | 张兆   | 0.54461296043                                                                                                       | nghuyong/ernie-3.0-base-zh | 30       | 10       | 2080Ti*2 batch=24<br />random_seed=42                                                                                                                                                 | best_1.pt : 0.9501953736953739<br /> best_2.pt : 0.8090291204084319<br />best_3.pt : 0.6498208126075585<br />best_4.pt : 0.9427773331560824<br />best_5.pt : 0.9111302191404046<br />best_6.pt : 0.8292753225364519<br />best_7.pt : 0.8938565221741899<br />best_8.pt : 0.9537918662275343<br />best_9.pt : 0.7997188727837208<br />best_10.pt : 0.8814895748089789<br />bagging : 0.954136902855915 | 0.616938<br />0.586474<br />0.529849<br />0.564196<br />0.568573<br />0.516865<br />0.46167<br />0.598384<br />0.542902<br />0.67397  |
| 训练开始：2022/10/13 11:39:52<br />训练结束：2022/10/13 16:40:56<br />提交时间：2022/10/13 17:38                                       | 张兆   | 0.56185948422                                                                                                       | nghuyong/ernie-3.0-base-zh | 50       | 10       | 2080Ti*2 batch=24<br />random_seed=42<br />correct_bias = True                                                                                                                        | best_1.pt : 0.958411<br />best_2.pt : 0.878444<br />best_3.pt : 0.950718<br />best_4.pt : 0.932414<br />best_5.pt : 0.929663<br />best_6.pt : 0.952183<br />best_7.pt : 0.952413<br />best_8.pt : 0.965563<br />best_9.pt : 0.949669<br />best_10.pt : 0.761528<br />bagging : 0.995802                                                                                                     | 0.627148<br />0.627045<br />0.524618<br />0.632453<br />0.603112<br />0.521837<br />0.550469<br />0.65892<br />0.498028<br />0.668365 |
| 训练开始：2022/10/15 10:00:01<br />训练结束：2022/10/15 10:25:30<br />训练开始：2022/10/15 11:25:11<br />训练结束：2022/10/15 11:59:54 | 张兆   | 未验证                                                                                                              | nghuyong/ernie-3.0-base-zh | 50       | 1        | 未添加RDrop：<br />2080Ti*2 batch=24<br />random_seed=42<br />correct_bias = True<br />添加RDrop：<br />2080Ti*4 batch=24<br />random_seed=42<br />correct_bias = True<br />RDrop=0.4 | 未添加RDrop：0.909574<br />添加RDrop：0.924773                                                                                                                                                                                                                                                                                                                                                        | 未添加RDrop：0.568337<br />添加RDrop：0.635372                                                                                        |
| 训练开始：2022/10/15 14:01:25<br />训练结束：2022/10/15 14:35:36                                                                       | 张兆   | 未验证                                                                                                              | nghuyong/ernie-3.0-base-zh | 50       | 1        | 添加FGM：<br />2080Ti*4 batch=24<br />random_seed=42<br />correct_bias = True                                                                                                         | 未添加FGM：0.909574<br />添加FGM：0.924612                                                                                                                                                                                                                                                                                                                                                            | 未添加FGM：0.568337<br />添加FGM：0.60684                                                                                             |
| 训练开始：2022/10/15 14:37:24<br />训练结束：2022/10/15 15:49:57                                                                       | 张兆   | 未验证                                                                                                              | nghuyong/ernie-3.0-base-zh | 50       | 1        | 添加PGD：<br />2080Ti*4 batch=24<br />random_seed=42<br />correct_bias = True<br />PGD_K=3                                                                                            | 未添加PGD：0.909574<br />添加PGD：0.917637                                                                                                                                                                                                                                                                                                                                                            | 未添加PGD：0.568337<br />添加PGD：0.582854                                                                                            |
| 训练开始：2022/10/15 21:18:01<br />训练结束：2022/10/16 03:22:04                                                                       | 张兆   | 0.56684304078                                                                                                       | nghuyong/ernie-3.0-base-zh | 50       | 10       | 2080Ti*4 batch=24<br />random_seed=42<br />correct_bias = True<br />RDrop=0.4                                                                                                         | best_1.pt : 0.964261<br />best_2.pt : 0.937494<br />best_3.pt : 0.93178<br />best_4.pt : 0.915917<br />best_5.pt : 0.954728<br />best_6.pt : 0.95125<br />best_7.pt : 0.951136<br />best_8.pt : 0.957213<br />best_9.pt : 0.959292<br />best_10.pt : 0.967787<br />bagging :  0.995802                                                                                                     | 0.58019<br />0.576124<br />0.526915<br />0.577535<br />0.612728<br />0.559396<br />0.549781<br />0.646989<br />0.537495<br />0.687826 |
|                                                                                                                                        | 李一鸣 | **0.60286913357**                                                                                             | Langboat/mengzi-bert-base  | 40       | 5        | batch=12 用官方测试集 F1 = 0.57 的模型产生的伪标签进行 pseudo-labelling                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                                                        | 李一鸣 | 0.587                                                                                                               | Langboat/mengzi-bert-base  | 40       | 1        | batch=12 用官方测试集 F1 = 0.57 的模型产生的伪标签进行 pseudo-labelling                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                                                        | 李一鸣 | 0.58719238                                                                                                          | Langboat/mengzi-bert-base  | 40       | 5        | batch=12, 用官方测试集 F1 = 0.60 的模型产生的伪标签进行 pseudo-labelling, RDrop=0.1                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                                                        | 李一鸣 | 0.58009964029                                                                                                       | Langboat/mengzi-bert-base  | 40       | 1        | batch=12, 用官方测试集 F1 = 0.60 的模型产生的伪标签进行 pseudo-labelling, RDrop=0.1                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                                                        | 李一鸣 | 0.60216                                                                                                             | Langboat/mengzi-bert-base  | 40       | 1        | batch=24, 用官方测试集 F1 = 0.599 的模型产生的伪标签(expand_train_cur_best.json )进行 pseudo-labelling, no extra tricks                                                               |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
| 训练开始：2022/10/16 17:28:14<br />训练结束：2022/10/16 18:10:27                                                                       | 张兆   | 0.52782175403                                                                                                       | Langboat/mengzi-bert-base  | 50       | 1        | 2080Ti*2 batch=24                                                                                                                                                                     | 0.911089                                                                                                                                                                                                                                                                                                                                                                                              | 0.604567                                                                                                                              |
| 训练开始：2022/10/16 18:10:31<br />训练结束：2022/10/16 18:58:40                                                                       | 张兆   | 0.50326362551                                                                                                       | Langboat/mengzi-bert-base  | 50       | 1        | 2080Ti*4 batch=24<br />RDrop=0.4                                                                                                                                                      | 0.900517                                                                                                                                                                                                                                                                                                                                                                                              | 0.560547                                                                                                                              |
| 训练开始：2022/10/16 18:58:44<br />训练结束：2022/10/16 19:41:36                                                                       | 张兆   | 0.50364182397                                                                                                       | nghuyong/ernie-3.0-base-zh | 50       | 1        | 2080Ti*2 batch=24                                                                                                                                                                     | 0.914422                                                                                                                                                                                                                                                                                                                                                                                              | 0.566295                                                                                                                              |
| 训练开始：2022/10/16 20:57:37<br />训练结束：2022/10/16 21:48:23                                                                       | 张兆   | 0.52432191354                                                                                                       | nghuyong/ernie-3.0-base-zh | 50       | 1        | 2080Ti*4 batch=24<br />RDrop=0.4                                                                                                                                                      | 0.924773                                                                                                                                                                                                                                                                                                                                                                                              | 0.635372                                                                                                                              |
|                                                                                                                                        | 张兆   | 0.60906089726<br />1：0.59323842065<br />3：**0.62876738448**<br />5：0.60725801303<br />2 3 4：0.61215379337 | nghuyong/ernie-3.0-base-zh | 40       | 5        | V100*4 batch=128<br />fgm                                                                                                                                                             | 0.940159<br />0.959345<br />0.982243<br />0.961593<br />0.961342<br />bagging:0.967881                                                                                                                                                                                                                                                                                                                | 0.83072<br />0.928769<br />0.927161<br />0.934107<br />0.943387                                                                       |

## 实验计划

探究不同因素对不同模型的影响

基本配置：batch=12，epoch=40 (early stop), gpu=2,3, lr=2e-5, seed=42, split_test_ratio=0.2, dropout=0.3

数据采用expand_train.json

| 模型                    | baseline                                                                       | rdrop=0.1                                                                     | rdrop=0.5 | rdrop=1.0                                                                     | ema=0.999                                                                     | pgd=3                                                                        | warmup=0.1                                                                    | fgm                                                                           |
| ----------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| mengzi-bert-base        | 2022_10_25_05_36_57<br />Epoch =11 0.943702<br />0.973459<br />0.58946182233  | 2022_10_25_05_36_51<br />Epoch =13 0.948241<br />0.981155<br />0.58861785176 |           |                                                                               | 2022_10_25_05_37_12<br />Epoch =12 0.953051<br />0.953051<br />0.58592248996 | 2022_10_25_05_37_15<br />Epoch =14 0.94846<br />0.982611<br />0.58877098445 | 2022_10_25_05_41_54<br />Epoch =20 0.955028<br />0.983924<br />0.58043132506 | 2022_10_25_05_37_08                                                           |
| ernie-3.0-base-zh       | 2022_10_21_17_44_31<br />Epoch = 12 0.939449<br />0.975266<br />0.61203747617 |                                                                               |           |                                                                               | 2022_10_22_16_39_14<br />Epoch=32 0.947126<br /> 0.986405<br /> 0.60179269740 | 2022_10_23_04_57_21<br />Epoch=20 0.94027<br />0.975243<br />0.58086732341   | 2022_10_21_21_46_43<br />Epoch =28 0.939117<br />0.981063<br />0.60027698592 | 2022_10_22_22_19_15<br /> Epoch=20 0.942571 <br />0.981936<br />0.59630101157 |
| chinese-macbert-base    | 2022_10_19_09_04_25<br />Epoch =11 0.93218<br />0.973964<br />0.58566511183   | 2022_10_21_17_25_43<br />Epoch =27 0.937448<br />0.984506<br />0.56675022857 |           | 2022_10_20_20_00_52<br />Epoch =19 0.944687<br />0.983283<br />0.57831287521 | 2022_10_22_03_38_29<br />Epoch =17 0.942641<br />0.981227<br />0.58781315810 | 2022_10_20_02_42_28<br />Epoch =19 0.950837<br />0.98362<br />0.59952852320 | 2022_10_21_04_24_30<br />Epoch =33 0.955382<br />0.990917<br />0.57818127106 | 2022_10_19_14_51_36<br />Epoch =25 0.940839<br />0.988269<br />0.59486694090 |
| chinese-roberta-wwm-ext |                                                                                |                                                                               |           |                                                                               |                                                                               |                                                                              |                                                                               |                                                                               |

## 英文模型

基本配置：batch=12，epoch=40 (early stop), gpu=2,3, lr=2e-5, seed=42, split_test_ratio=0.2, dropout=0.3

数据采用expand_train_cur_best_en.json

| 模型              | baseline                                                                      | ema=0.999                                                                     | pgd=3                                                                         | fgm                                                                           | warmup=0.1                                                                    |
| ----------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| ernie-2.0-base-en | 2022_10_23_18_05_14<br />Epoch =24 0.869316<br />0.947128<br />0.58633437923 | 2022_10_23_18_18_01<br />Epoch =27 0.883606<br />0.973969<br />0.58900858108 | 2022_10_23_18_05_27<br />Epoch =18 0.851685<br />0.94073<br />0.56777475031  | 2022_10_23_18_19_40<br />Epoch =24 0.884273<br />0.953767<br />0.60969211560 | 2022_10_23_18_05_32<br />Epoch =25 0.86958<br />0.968817<br />0.59526836713  |
| roberta-base      |                                                                               |                                                                               |                                                                               |                                                                               |                                                                               |
| xlnet-base-cased  | 2022_10_24_05_41_37<br />Epoch =31 0.876336<br />0.976078<br />0.59146824680 | 2022_10_24_05_41_04<br />Epoch =17 0.868918<br />0.969398<br />0.59067118434 | 2022_10_24_05_41_12<br />Epoch =19 0.875453<br />0.970272<br />0.57834073199 | 2022_10_24_05_41_20<br />Epoch =21 0.879417<br />0.976025<br />0.58459104336 | 2022_10_24_05_41_29<br />Epoch =24 0.873381<br />0.974582<br />0.59797619630 |
| deberta-v3-base   | 2022_10_24_10_44_57<br />Epoch =25 0.875878<br />0.943788<br />0.58797673621 | 2022_10_24_10_50_12<br />Epoch =37 0.880665<br />0.947626<br />0.59859900583 | 2022_10_24_10_52_08<br />Epoch =28 0.894997<br />0.955026<br />0.60094762341 | 2022_10_24_10_54_03<br />Epoch =24 0.881334<br />0.947271<br />0.59791066113 | 2022_10_24_10_51_34<br />Epoch =24 0.856213<br />0.933411<br />0.58096981486 |

ernie-2.0-base-en+fgm 十折交叉验证 expand_train_cur_best_en.json

2022_10_25_05_36_32

ernie-3.0-base-zh+fgm 十折交叉验证 expand_train_cur_best+en_zh.json

2022_10_25_05_36_40

**0.62529548458**
