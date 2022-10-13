# CCF-BDCI 小样本数据分类任务

## 参数设置

```python
parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--predict', action='store_true', help='Whether to predict')

parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--datetime',type=str, required=True, help='Get Time Stamp')
parser.add_argument('--epoch', type=int, default=5, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')

parser.add_argument('--data_folder_dir',type=str, required=True, help='Data Folder Location')
parser.add_argument('--data_file',type=str, help='Data Filename')

parser.add_argument('--checkpoint',type=int, default=0, help='Use checkpoint')
parser.add_argument('--load', action='store_true', help='load from checkpoint')
parser.add_argument('--load_pt', type=str, help='load from checkpoint')
parser.add_argument('--save', action='store_true', help='Whether to save model')

parser.add_argument('--bert',type=str, required=True, help='Choose Bert')
parser.add_argument('--K', type=int, default=1, help='K-fold')
parser.add_argument('--warmup',type=float, default=0.1, help='warm up ratio')

args = parser.parse_args()
```

## 训练

```bash
python main.py \
--train \
--batch 24 --board --datetime ${TIMESTAMP} --epoch 50 --gpu 2 3 --lr 2e-5 --seed ${SEED} \
--data_folder_dir fewshot --data_file train.json \
--checkpoint 10 --save \
--bert ${BERT} --K ${K}
```

## 测试

```bash
python main.py \
--test \
--batch 512 --datetime ${TIMESTAMP} --gpu 2 3 --seed ${SEED} \
--data_folder_dir fewshot --data_file train.json \
--bert ${BERT} --K ${K}
```

## 推理

```bash
python main.py \
--predict \
--batch 512  --datetime ${TIMESTAMP} --gpu 2 3 --seed ${SEED} \
--data_folder_dir fewshot --data_file testA.json \
--bert ${BERT} --K ${K}
```

## 打包

```bash
python pack.py --datetime 2022_10_13_10_59_28 --score 0.0001245
```

# 训练记录

## TO DO LIST

- [X] AdamW  加 correct_bias = True 《Revisiting Few-sample BERT Fine-tuning》 https://zhuanlan.zhihu.com/p/524036087
- [ ] 对官方测试集打伪标签加入训练集

## 记录

| 时间                                                                                             | 成员 | 得分                    | 预训练模型                 | 训练轮数 | 交叉验证 | 其他设置                                                              | 训练集+验证集得分                                                                                                                                                                                                                                                                                                                                                                                     | 验证集得分                                                                                                                            |
| ------------------------------------------------------------------------------------------------ | ---- | ----------------------- | -------------------------- | -------- | -------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 训练开始：2022/10/12 15:49:35<br />训练结束：2022/10/12 18:52:39<br />提交时间：2022/10/12 18:54 | 张兆 | 0.46661411496           | nghuyong/ernie-3.0-base-zh | 60       | 5        | warmup 0.1（用的可能不对）<br />2080Ti*2 batch=24<br />random_seed=42 | best_1.pt : 0.7630403185746878<br />best_2.pt : 0.785194086089213<br />best_3.pt : 0.8484339825318716<br />best_4.pt : 0.8481442706893945<br />best_5.pt : 0.8564035464087444<br />bagging : 0.9039802357258913                                                                                                                                                                                       | 0.474196<br />0.436003<br />0.444065<br />0.501953<br />0.502899                                                                      |
| 训练开始：2022/10/12 20:34:47<br />训练结束：2022/10/12 20:52:25<br />提交时间：2022/10/12 20:54 | 张兆 | 0.50926177              | Langboat/mengzi-bert-base  | 25       | 1        | 2080Ti*2 batch=24<br />random_seed=42<br />random_split_ratio=0.8     | 0.9035649164096997                                                                                                                                                                                                                                                                                                                                                                                    | 0.555419                                                                                                                              |
| 训练开始：2022/10/12 20:57:30<br />训练结束：2022/10/13 00:12:28<br />提交时间：2022/10/13 08:23 | 张兆 | **0.54461296043** | nghuyong/ernie-3.0-base-zh | 30       | 10       | 2080Ti*2 batch=24<br />random_seed=42                                 | best_1.pt : 0.9501953736953739<br /> best_2.pt : 0.8090291204084319<br />best_3.pt : 0.6498208126075585<br />best_4.pt : 0.9427773331560824<br />best_5.pt : 0.9111302191404046<br />best_6.pt : 0.8292753225364519<br />best_7.pt : 0.8938565221741899<br />best_8.pt : 0.9537918662275343<br />best_9.pt : 0.7997188727837208<br />best_10.pt : 0.8814895748089789<br />bagging : 0.954136902855915 | 0.580996<br />0.525019<br />0.518575<br />0.560631<br />0.514446<br />0.470862<br />0.429101<br />0.598384<br />0.456735<br />0.57741 |
| 训练开始：2022/10/13 11:39:52<br />训练结束：<br />提交时间：                                    | 张兆 |                         | nghuyong/ernie-3.0-base-zh | 60       | 10       | 2080Ti*2 batch=24<br />random_seed=42<br />correct_bias = True        |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
|                                                                                                  |      |                         |                            |          |          |                                                                       |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                       |
