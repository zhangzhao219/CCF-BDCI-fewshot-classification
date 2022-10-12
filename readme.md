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