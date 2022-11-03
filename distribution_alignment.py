import numpy as np
import pandas as pd

def norm(x):
    return x / np.sum(x, axis=-1).reshape(-1, 1)

# train.json 的分布
p = norm(np.array([33, 19, 183, 33, 44, 36, 47, 48, 39, 25, 52, 54, 7, 16, 19, 33, 17, 13, 16, 13, 25, 12, 5, 22, 29, 16, 17, 22, 8, 7, 13, 4, 5, 8, 12, 6]))

# 均匀分布
balance_p = norm(np.array([1 / 36 for _ in range(36)]))

# 模型预测分数
score_df = pd.read_csv("goodresult/0.6255.csv")
score_mat = np.array(score_df.values[:, 2:-1])

q = score_mat
# 用所有数据预测向量的平均数估计 波浪线 p
p_hat = score_mat.mean(axis = 0)

# 向训练集分布对齐

da_score = norm(q * (p / p_hat))

# 向均匀分布对齐 (可以一定程度上增大 tail class 的分数)

balance_score = norm(q * (balance_p / p_hat))

# pd.DataFrame(da_score.cpu().numpy()).to_csv("tmp.csv")
alpha, beta, gamma = 0.9, 0, 0.1

# 加权得到最终分数
aggregate_score = alpha * q + beta * da_score + gamma * balance_score
# pred_label = aggregate_score.max(axis = -1)[1]

# pred_ms = aggregate_score.max(axis = -1)[0]
# print(pred_ms, pred_label)
score_df['label_1'] = aggregate_score.argmax(axis=1)


score_df['label_2'] = np.argmax(q*norm(1-p),axis=1)


score_df['labelo'] = ((score_df['label_1'] != score_df['label']) | (score_df['label_1'] != score_df['label_2']) | (score_df['label'] != score_df['label']))
print(score_df.loc[score_df['labelo'] == True,['label','label_1','label_2']])
# score_df['labelo'] = ((score_df['label_1'] == score_df['label_2']) & (score_df['label_1'] != score_df['label']))
# print(score_df.loc[score_df['labelo'] == True,['label','label_1','label_2']])
print(len(score_df.loc[score_df['labelo'] == True,['label','label_1','label_2']]))

print(sum(score_df['label_1'] != score_df['label']))
print(sum(score_df['label_2'] != score_df['label']))
print(sum(score_df['label_2'] != score_df['label_1']))