# # all parameters
# python main.py \
# --batch \
# --board \
# --checkpoint \
# --data_dir \
# --epoch \
# --gpu \
# --K \
# --load \
# --load_pt \
# --lr \
# --predict \
# --save \
# --seed \
# --test \
# --train \

# # train and predict with K-Fold
# python main.py \
# --train --predict \
# --batch 8 \
# --epoch 3 \
# --data_dir data/ag_news_csv2 \
# --K 5 \
# --checkpoint 10 \
# --board --save --gpu 0 --lr 0.001 --seed 42 

# # train and test without K-Fold
# python main.py \
# --train --test\
# --batch 8 \
# --epoch 3 \
# --data_dir data/ag_news_csv2 \
# --checkpoint 10 \
# --board --save --gpu 0 --lr 0.001 --seed 42

# # load model
# --load checkpoint_0_epoch.pt \

# python3 main.py \
# --train \
# --batch 24 \
# --epoch 5 \
# --data_dir fewshot \
# --checkpoint 20 \
# --K 1 \
# --save --gpu 2 3 --lr 2e-5 --seed 42


## 训练

python3 main.py \
--train \
--batch 24 \
--epoch 60 \
--data_dir fewshot \
--checkpoint 10 \
--K 5 \
--save --gpu 2 3 --lr 2e-5 --seed 42


## 测试

python3 main.py \
--test \
--batch 256 \
--data_dir fewshot \
--K 5 \
--gpu 2 3 --lr 2e-5 --seed 42

## 推理

python3 main.py \
--predict \
--batch 256 \
--data_dir fewshot \
--K 5 \
--gpu 2 3 --lr 2e-5 --seed 42