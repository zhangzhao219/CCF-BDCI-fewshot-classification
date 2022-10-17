import os
import json
import argparse
import pandas as pd
import numpy as np

DATA_PATH = './data/'
num_labels = 36

parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--predict_csv', type=str, required=True, help='Result (with confidence score) Location')
parser.add_argument('--corpus_json', type=str, required=True, help='Test json file')
parser.add_argument('--origin_train_json', type=str, required=True, help='Train json file')
parser.add_argument('--data_folder_dir', type=str, required=True, help='Data Folder Location')
parser.add_argument('--fix_thresh', type=float, default=0.70, help='Fixed thresh')
parser.add_argument('--percent', type=float, default=50, help='top (100% - percent) confidence score will be regarded as reliable pseudo labels for a given class')

args = parser.parse_args()

def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r", encoding='UTF-8') as f:
        reader = f.readlines()
    return [json.loads(line.strip()) for line in reader]

df = pd.read_csv(args.predict_csv)
pseudo_label_df = pd.DataFrame()
class_thresh = []

for l in range(num_labels):
    sub_df = df[df['label'] == l]
    sub_score = np.array(sub_df['score'])
    thresh = np.percentile(sub_score, args.percent)
    if thresh < args.fix_thresh:
        thresh = args.fix_thresh
    sub_df = sub_df.drop(sub_df[sub_df['score'] < thresh].index)
    pseudo_label_df = pd.concat([pseudo_label_df, sub_df])

corpus_df = pd.DataFrame.from_records(read_json((os.path.join(DATA_PATH, args.data_folder_dir, args.corpus_json))))
corpus_df['label_id'] = -1

for idx, row in pseudo_label_df.iterrows():
    id = row['id']
    corpus_df.loc[corpus_df['id'] == id, 'label_id'] = row['label']
corpus_df = corpus_df.drop(corpus_df[corpus_df['label_id'] == -1].index)

origin_train_df = pd.DataFrame.from_records(read_json(os.path.join(DATA_PATH, args.data_folder_dir, args.origin_train_json)))
expand_df = pd.concat([origin_train_df, corpus_df])
expand_json = expand_df.to_json(orient="records",force_ascii=False)
expand_json = json.loads(expand_json)

with open(os.path.join(DATA_PATH, args.data_folder_dir, 'expand_train.json'), 'w', encoding='utf-8') as f:
    for item in expand_json:
        line = json.dumps(item, ensure_ascii=False)
        f.write(line + '\n')

f.close()
