# 镜像复现流程说明

## 镜像准备

首先对压缩的镜像进行解压缩

```bash
tar xvzf image/IMAGE.tar.gz -C image
```

然后会得到下面的文件夹结构：

```bash
.
├── data
│   ├── README.md
│   ├── code
│   │   ├── inference
│   │   │   ├── inference_submit.sh
│   │   │   └── inference_submit_after_train.sh
│   │   ├── own_code
│   │   │   ├── add_pseudo_labels.py
│   │   │   ├── back_trans.py
│   │   │   ├── dataset.py
│   │   │   ├── eda.py
│   │   │   ├── ensemble.py
│   │   │   ├── log.py
│   │   │   ├── main.py
│   │   │   ├── model1.py
│   │   │   ├── pretrained
│   │   │   │   └── nghuyong
│   │   │   │       └── ernie-3.0-base-zh
│   │   │   │           ├── config.json
│   │   │   │           ├── pytorch_model.bin
│   │   │   │           └── vocab.txt
│   │   │   └── trick.py
│   │   ├── requirements.txt
│   │   └── train
│   │       ├── 2022_10_22_19_12_04-3-0.62876738448.sh
│   │       ├── 2022_10_27_07_38_29_3-0.63077875449-8.sh
│   │       ├── 2022_11_01_04_26_32-3-0.63293263685.sh
│   │       ├── 2022_11_03_19_41_25-a-0.63234589689.sh
│   │       ├── 2022_11_05_05_55_17-0-0.62600679310.sh
│   │       ├── 2022_11_06_04_35_15-a-0.62673116125.sh
│   │       └── 2022_11_06_19_08_24-a-.sh
│   ├── prediction_result
│   ├── raw_data
│   │   ├── testA.json
│   │   └── testB.json
│   └── user_data
│       ├── log
│       ├── models
│       │   ├── 2022_10_22_19_12_04-3-0.62876738448
│       │   │   └── best_3.pt_half.pt
│       │   ├── 2022_10_27_07_38_29_3-0.63077875449
│       │   │   └── best_3.pt_half.pt
│       │   ├── 2022_11_01_04_26_32-3-0.63293263685
│       │   │   └── best_3.pt_half.pt
│       │   ├── 2022_11_03_19_41_25-a-0.63234589689
│       │   │   ├── best_1.pt_half.pt
│       │   │   └── best_2.pt_half.pt
│       │   ├── 2022_11_05_05_55_17-0-0.62600679310
│       │   │   └── best_0.pt_half.pt
│       │   ├── 2022_11_06_04_35_15-a-0.62673116125
│       │   │   ├── best_1.pt_half.pt
│       │   │   └── best_2.pt_half.pt
│       │   └── 2022_11_06_19_08_24-a-
│       │       └── best_2.pt_half.pt
│       ├── models_after_train
│       └── pseudo_data
│           ├── eda_data.json
│           ├── expand_train_6422_aug_tail.json
│           ├── expand_train_6460_aug_tail.json
│           ├── expand_train_aug_tail.json
│           ├── expand_train_cur_best.json
│           └── train_632_aug_tail.json
├── image
│   ├── Final.tar
│   ├── IMAGE.tar.gz
│   └── README.md
├── run_inference.sh
├── run_inference_after_train.sh
└── run_train.sh
```

## 环境要求

Linux系统

Docker（root权限）

CUDA 11.4 或 11.2

GPU（具体型号见下方）

## 直接推理

```bash
sudo bash run_inference.sh
```

会调用 `data/user_data/models`内部的9个模型进行推理，输出的结果文件在 `data/prediction_result`中，名称为 `finalB.csv`

**由于推理使用的机器可能不同，结果可能有一点点差异**

提交版本是 `NVIDIA GeForce RTX 2080 Ti 11G * 2`进行推理后得到的结果，如果在 `NVIDIA Tesla V100 32G * 4`上进行推理，最终结果有两条预测不同

## 训练

```bash
sudo bash run_train.sh
```

训练注意事项：

1. 如果GPU的数量足够，`run_train.sh`内部的7个脚本可以并行运行，更改GPU卡号在 `data/code/train`内部脚本的第四行 `GPU='0 1 2 3'`
2. 训练过程大部分是在 `NVIDIA Tesla V100 32G * 4`上进行的，其中第2个脚本 `2022_10_27_07_38_29_3-0.63077875449-8.sh`是在 `NVIDIA Tesla V100 32G * 8`上进行的。也就是说，训练的硬件条件最少需要 `NVIDIA Tesla V100 32G * 8`
3. 训练后的模型存放在 `data/user_data/models_after_train`中，**注意虽然生成的模型数量大于9，但是在后面推理的时候仅用到了其中的9个模型**
4. 有输出日志，存放在 `data/user_data/log`内部

## 使用训练后的模型进行推理

```bash
sudo bash run_inference_after_train.sh
```

会调用 `data/user_data/models_after_train`内部的9个模型进行推理，调用的9个模型名称写在 `data/code/inference/inference_submit_after_train.sh`内部，输出的结果文件在 `data/prediction_result`中，名称为 `finalB.csv`
