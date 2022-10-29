# Variables
SEED=42
K=1
GPU='2 3'
TEST_BATCH=128
TRAIN_FILE=train.json
LABEL=36
DROPOUT=0.3
MODEL=model1
# 直接联网下载模型文件
# BERT='nghuyong/ernie-3.0-base-zh'
# 调用已经下载好的原始文件
BERT='pretrained/nghuyong/ernie-3.0-base-zh'
TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S)

echo $TIMESTAMP

nvidia-smi

## 训练
python main.py \
--train \
--batch 12 --board --datetime ${TIMESTAMP} --epoch 1 --gpu ${GPU} --lr 2e-5 --seed ${SEED} --early_stop 10 \
--data_folder_dir fewshot --data_file ${TRAIN_FILE} --label ${LABEL} \
--checkpoint 20 --save \
--bert ${BERT} --dropout ${DROPOUT} --feature_layer 4 --freeze 0 --model ${MODEL} \
--K ${K} --split_test_ratio 0.2
# --en \
# --awp 1 --ema 0.999 --fgm --fl --mixif --pgd 3 --rdrop 0.1 --sce --swa --warmup 0.1
  
## 测试
python main.py \
--test \
--batch ${TEST_BATCH} --datetime ${TIMESTAMP} --gpu ${GPU} \
--data_folder_dir fewshot --data_file ${TRAIN_FILE} --label ${LABEL} \
--bert ${BERT} --dropout ${DROPOUT} --feature_layer 4 --freeze 0 --model ${MODEL} \
--K ${K}
# --en \


## 推理
python main.py \
--predict \
--batch ${TEST_BATCH}  --datetime ${TIMESTAMP} --gpu ${GPU} \
--data_folder_dir fewshot --data_file testA.json --label ${LABEL} \
--bert ${BERT} --dropout ${DROPOUT} --feature_layer 4 --freeze 0 --model ${MODEL} \
--K ${K}
# predict_with_score \
# --en \
