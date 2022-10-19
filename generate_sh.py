import os

with open('run.sh','w') as f:
    for i in os.listdir('./runsh'):
        for j in os.listdir('./runsh/' + i):
            f.writelines('bash ./runsh/' + i + '/' + j + '\n')