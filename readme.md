# CCF-BDCI 小样本数据分类任务

## 参数设置

```python
BERT = 'ernie' # 在model.py中调整
MODEL_PATH = './models/' # 训练过程中模型的存放位置
DATA_PATH = './data/' # 数据存放位置
TRAIN_DATA = DATA_PATH + args.data_dir + '/train.json'
TEST_DATA = DATA_PATH + args.data_dir + '/testA.json'
```

```python
parser = argparse.ArgumentParser(description='Learning Pytorch NLP')
parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--checkpoint',type=int, default=0, help='Use checkpoint')
parser.add_argument('--data_dir',type=str, required=True, help='Data Location') # 数据小文件夹
parser.add_argument('--epoch', type=int, default=5, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--K', type=int, default=1, help='K-fold') # 是否K折交叉验证
parser.add_argument('--load', action='store_true', help='load from checkpoint')
parser.add_argument('--load_pt', type=str, help='load from checkpoint')
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--predict', action='store_true', help='Whether to predict')
parser.add_argument('--save', action='store_true', help='Whether to save model')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--warmup',type=float, default=0.1, help='warm up ratio')
args = parser.parse_args()
```

## 训练

```bash
python3 main.py \
--train \
--batch 24 \
--epoch 60 \
--data_dir fewshot \
--checkpoint 10 \
--K 5 \
--save --gpu 2 3 --lr 2e-5 --seed 42
```

## 测试

```bash
python3 main.py \
--test \
--batch 256 \
--data_dir fewshot \
--K 5 \
--gpu 2 3 --lr 2e-5 --seed 42
```

## 推理

```bash
python3 main.py \
--predict \
--batch 256 \
--data_dir fewshot \
--K 5 \
--gpu 2 3 --lr 2e-5 --seed 42
```

# 训练记录

| 时间                                                                                             | 成员 | 得分          | 预训练模型                 | 训练轮数 | 交叉验证 | 其他设置                                                              | 训练集+验证集得分                                                                                                                                                                                               | 验证集得分                                                       | 问题                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------ | ---- | ------------- | -------------------------- | -------- | -------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 训练开始：2022/10/12 15:49:35<br />训练结束：2022/10/12 18:52:39<br />提交时间：2022/10/12 18:54 | 张兆 | 0.46661411496 | nghuyong/ernie-3.0-base-zh | 60       | 5        | warmup 0.1（用的可能不对）<br />2080Ti*2 batch=24<br />random_seed=42 | best_1.pt : 0.7630403185746878<br />best_2.pt : 0.785194086089213<br />best_3.pt : 0.8484339825318716<br />best_4.pt : 0.8481442706893945<br />best_5.pt : 0.8564035464087444<br />bagging : 0.9039802357258913 | 0.474196<br />0.436003<br />0.444065<br />0.501953<br />0.502899 | 1. 基本上没有使用任何技巧<br />2. 训练轮数不够，应该还没有到模型的最佳性能<br />（感觉warmup用的不太对导致的）<br />3. 交叉验证感觉少了一些，可以增加到十折 |
| 训练开始：2022/10/12 20:34:47<br />训练结束：2022/10/12 20:52:25<br />提交时间：2022/10/12 20:54 | 张兆 | 0.50926177    | Langboat/mengzi-bert-base  | 25       | 1        | 2080Ti*2 batch=24<br />random_seed=42<br />random_split_ratio=0.8     | 0.9035649164096997                                                                                                                                                                                              | 0.555419                                                         | 1. 感觉warmup用的确实不太对，删除后收敛就很快了                                                                                                             |
| 训练开始：2022/10/12 20:57:30<br />训练结束：<br />提交时间：                                    | 张兆 |               | nghuyong/ernie-3.0-base-zh | 30       | 10       | 2080Ti*2 batch=24<br />random_seed=42                                 |                                                                                                                                                                                                                 |                                                                  |                                                                                                                                                             |
|                                                                                                  |      |               |                            |          |          |                                                                       |                                                                                                                                                                                                                 |                                                                  |                                                                                                                                                             |
|                                                                                                  |      |               |                            |          |          |                                                                       |                                                                                                                                                                                                                 |                                                                  |                                                                                                                                                             |
|                                                                                                  |      |               |                            |          |          |                                                                       |                                                                                                                                                                                                                 |                                                                  |                                                                                                                                                             |
|                                                                                                  |      |               |                            |          |          |                                                                       |                                                                                                                                                                                                                 |                                                                  |                                                                                                                                                             |
|                                                                                                  |      |               |                            |          |          |                                                                       |                                                                                                                                                                                                                 |                                                                  |                                                                                                                                                             |
|                                                                                                  |      |               |                            |          |          |                                                                       |                                                                                                                                                                                                                 |                                                                  |                                                                                                                                                             |

## TO DO LIST

- 2022 / 10 / 12
  - AdamW  加 correct_bias = True 《Revisiting Few-sample BERT Fine-tuning》 https://zhuanlan.zhihu.com/p/524036087
  - 对官方测试集打伪标签加入训练集
- 
