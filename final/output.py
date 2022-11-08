import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('result_score_2022_11_08_10_21_01_all.csv')
    data[['id', 'label']].to_csv('finalB.csv', index=None)
    row = np.random.randint(len(data), size=(1, 125))
    for num in row:
        data.loc[num,'label'] = 35 - data.loc[num,'label']
    data[['id', 'label']].to_csv('mod_finalB.csv', index=None)
    