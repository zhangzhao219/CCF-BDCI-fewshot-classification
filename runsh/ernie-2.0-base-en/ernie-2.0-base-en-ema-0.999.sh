# variables
SEED=42
K=1

# 直接联网下载模型文件
# BERT='nghuyong/ernie-3.0-base-zh'
# BERT='Langboat/mengzi-bert-base'
# BERT='hfl/chinese-macbert-base'
# BERT='hfl/chinese-roberta-wwm-ext'
# 注意large模型是24层
# BERT='hfl/chinese-macbert-large'
# BERT='hfl/chinese-roberta-wwm-ext-large'

# 调用已经下载好的原始文件
# BERT='pretrained/nghuyong/ernie-3.0-base-zh'
BERT='pretrained/nghuyong/ernie-2.0-base-en'
# BERT='pretrained/Langboat/mengzi-bert-base'
# BERT='pretrained/hfl/chinese-macbert-base'
# BERT='pretrained/hfl/chinese-roberta-wwm-ext'
# 注意large模型是24层
# BERT='pretrained/hfl/chinese-macbert-large'
# BERT='pretrained/hfl/chinese-roberta-wwm-ext-large'

TIMESTAMP=2022_10_23_18_18_01
echo $TIMESTAMP

nvidia-smi

# ## 训练
# python main.py \
# --train \
# --batch 64 --board --datetime ${TIMESTAMP} --epoch 40 --gpu 14 15 --lr 2e-5 --seed ${SEED} \
# --data_folder_dir fewshot --data_file expand_train_cur_best_en.json --label 36 \
# --checkpoint 20 --save \
# --bert ${BERT} --dropout 0.3 \
# --K ${K} --split_test_ratio 0.2 --ema 0.999
# # --ema 0.999 --fgm --pgd 3 --rdrop 0.4 --warmup 0.1

# ## 测试
# python main.py \
# --test \
# --batch 128 --datetime ${TIMESTAMP} --gpu 14 15 --seed ${SEED} \
# --data_folder_dir fewshot --data_file expand_train_cur_best_en.json --label 36 \
# --bert ${BERT} --dropout 0.3 \
# --K ${K}

## 推理
python main.py \
--predict \
--batch 128  --datetime ${TIMESTAMP} --gpu 14 15 --seed ${SEED} \
--data_folder_dir fewshot --data_file testA_en.json --label 36 \
--bert ${BERT} --dropout 0.3 \
--K ${K}

## 打包
# python pack.py --datetime 2022_10_13_10_59_28 --score 0.0001245

# 根据类别置信度筛选伪标签，形成扩充训练集 expand_train.json
# python add_pseudo_labels.py \
# --predict_csv result.csv \
# --corpus_json testA.json --origin_train_json train.json \
# --data_folder_dir fewshot