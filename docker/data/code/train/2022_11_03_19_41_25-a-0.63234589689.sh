# Variables
SEED=42
K=4
GPU='0 1 2 3'
TEST_BATCH=256
TRAIN_FILE=train_632_aug_tail.json
LABEL=36
DROPOUT=0.3
MODEL=model1

BERT='../own_code/pretrained/nghuyong/ernie-3.0-base-zh'
TIMESTAMP=2022_11_03_19_41_25

echo $TIMESTAMP

nvidia-smi

## 训练
python ../own_code/main.py \
--train \
--batch 128 --datetime ${TIMESTAMP} --epoch 50 --gpu ${GPU} --lr 2e-5 --seed ${SEED} --early_stop 10 \
--data_folder_dir ../../user_data/pseudo_data/ --data_file ${TRAIN_FILE} --label ${LABEL} \
--save \
--bert ${BERT} --dropout ${DROPOUT} --feature_layer 4 --freeze 0 --model ${MODEL} \
--K ${K} --split_test_ratio 0.2 --fgm --K_stop 2
