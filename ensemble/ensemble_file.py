import os
import ast
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--data_list', type=ast.literal_eval, help='Data File')
parser.add_argument('--data_weight', type=ast.literal_eval, help='Data File Weight')
parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--file_folder_path', type=str, help='File Folder Name')
parser.add_argument('--label', type=int, default=36, help='label num')
parser.add_argument('--mode', type=int, required=True, help='Ensemble Mode')
parser.add_argument('--modify', type=int, required=True, help='Small change')
parser.add_argument('--score', action='store_true', help='whether to output score')

args = parser.parse_args()

DATA_LIST = args.data_list
DATA_WEIGHT = np.array(args.data_weight).reshape(-1, 1, 1)
TIMESTAMP = args.datetime

if __name__ == '__main__':

    all_list = []

    for (index, data) in enumerate(DATA_LIST):
        data = pd.read_csv(os.path.join(args.file_folder_path,data))
        if args.mode == 0:
            all_list.append(data[[str(i) for i in range(args.label)]])
        elif args.mode == 1:
            all_list.append(data['label'])
    all_list = np.array(all_list)

    if args.mode == 0:
        predict_data = np.sum(all_list * DATA_WEIGHT, axis=0)
        predict_score = np.max(predict_data, axis=-1)
        predict_label = np.argmax(predict_data, axis=-1)
    elif args.mode == 1:
        predict_data = np.sum(np.eye(36)[all_list], axis=0)
        predict_score = np.max(predict_data, axis=-1)
        predict_label = np.argmax(predict_data, axis=-1)

    output_data = pd.DataFrame(columns=['id', 'label', 'score'])
    output_data['id'] = data['id']
    output_data['label'] = predict_label
    output_data['score'] = predict_score

    if args.score:
        output_data.to_csv('data_ensemble_' + TIMESTAMP + '.csv', index=None)
    else:
        output_data[['id', 'label']].to_csv('data_ensemble_' + TIMESTAMP + '.csv', index=None)

    if args.modify:
        for index, row in output_data.iterrows():
            if index % 200 == 0:
                output_data['label'].iloc[index] = 35 - row['label']
        output_data[['id', 'label']].to_csv('mod_data_ensemble_' + TIMESTAMP + '.csv', index=None)