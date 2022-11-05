# 伪标签产生历程

- expand_train_1.json

  结果文件：result_2022-10-13-15-15.csv (test F1 = 0.51443849647)

  产生结果文件的实验配置：K=1, batch=16, bert='mengzi', board=False, checkpoint=20, data_file='train.json', data_folder_dir='fewshot', datetime='2022-10-13-15-15', epoch=50, gpu=['0'], load=False, load_pt=None, lr=2e-05, predict=False, pseudo=False, save=True, seed=42, test=False, train=True, warmup=0.1

  伪标签的实验配置：percentile = 85, fix_thresh = 0.70

- expand_train_2.json

  结果文件：result_2022-10-13-22-48.csv (test F1 = 0.55291467089)

  产生结果文件的实验配置：K=1, batch=16, bert='mengzi', board=False, checkpoint=20, data_file='expand_train_1.json', data_folder_dir='fewshot', datetime='2022-10-13-22-48', epoch=50, gpu=['0'], load=False, load_pt=None, lr=2e-05, predict=False, pseudo=False, save=True, seed=42, test=False, train=True, warmup=0.1

  伪标签的实验配置：percentile = 70, fix_thresh = 0.70

- expand_train_3.json

  结果文件：result_2022-10-13-23-41.csv (test F1 = 0.56821691118)

  产生结果文件的实验配置：K=1, batch=16, bert='mengzi', board=False, checkpoint=20, data_file='expand_train_2.json', data_folder_dir='fewshot', datetime='2022-10-13-23-41', epoch=50, gpu=['0'], load=False, load_pt=None, lr=2e-05, predict=False, pseudo=False, save=True, seed=42, test=False, train=True, warmup=0.1

  伪标签的实验配置：percentile = 50, fix_thresh = 0.70

- expand_train_4.json

  结果文件：result_2022-10-13-23-41.csv (test F1 = 0.56821691118)

  产生结果文件的实验配置：(K=1, batch=16, bert='mengzi', board=False, checkpoint=20, data_file='expand_train_2.json', data_folder_dir='fewshot', datetime='2022-10-13-23-41', epoch=50, gpu=['0'], load=False, load_pt=None, lr=2e-05, predict=False, pseudo=False, save=True, seed=42, test=False, train=True, warmup=0.1)

  伪标签的实验配置：percentile = 30, fix_thresh = 0.70

- expand_train.json

  结果文件：result_2022-10-14-13-52.csv (test F1 = 0.60286913357)

  产生结果文件的实验配置：(K=5, batch=32, bert='mengzi', board=False, checkpoint=40, data_file=expand_train_4.json, data_folder_dir='fewshot', datetime='2022-10-14-13-52', epoch=50, gpu=['0'], load=False, load_pt=None, lr=2e-05, predict=False, pseudo=False, save=True, seed=42, test=False, train=True, warmup=0.1),

  伪标签的实验配置：percentile = 25, fix_thresh = 0.70

- expand_train_cur_best.json:

  结果文件： result_score_2022_10_20_02_42_28.csv (test F1 = 0.60286913357) 

  产生结果文件的实验配置：(chinese-macbert-base, expand_train.json, batch=12，epoch=40 (early stop), gpu=2,3, lr=2e-5, seed=42, split_test_ratio=0.2, dropout=0.3) , percentile = 20, fix_thresh = 0.70

- expand_train_cur_best_aug_tail.json

  expand_train_cur_best.json ＋ 尾部类别 (12, 22, 32, 35) 大语种翻译 (每句话 5 次) + Chinese EDA 数据增强 (每句话 20 次)

- expand_train_630_aug_tail.json

  test F1 = 0.6307787544 的模型打伪标签（percentile = 20, fix_thresh = 0.70）＋ 尾部类别 (12, 22, 32, 35) 大语种翻译 (每句话 5 次) + Chinese EDA 数据增强 (每句话 20 次)

- expand_train_632_aug_tail.json

  test F1 = 0.63293263685 的模型打伪标签（percentile = 20, fix_thresh = 0.70） ＋ 尾部类别 (12, 22, 32, 35) 大语种翻译 (每句话 5 次) + Chinese EDA 数据增强 (每句话 20 次)

