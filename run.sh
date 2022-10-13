# variables
SEED=42
K=10
BERT=ernie
TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S)
echo $TIMESTAMP


## 训练
python main.py \
--train \
--batch 24 --board --datetime ${TIMESTAMP} --epoch 50 --gpu 2 3 --lr 2e-5 --seed ${SEED} \
--data_folder_dir fewshot --data_file train.json \
--checkpoint 10 --save \
--bert ${BERT} --K ${K}
  
## 测试
python main.py \
--test \
--batch 512 --datetime ${TIMESTAMP} --gpu 2 3 --seed ${SEED} \
--data_folder_dir fewshot --data_file train.json \
--bert ${BERT} --K ${K}

## 推理
python main.py \
--predict \
--batch 512  --datetime ${TIMESTAMP} --gpu 2 3 --seed ${SEED} \
--data_folder_dir fewshot --data_file testA.json \
--bert ${BERT} --K ${K}

## 打包
# python pack.py --datetime 2022_10_13_10_59_28 --score 0.0001245
