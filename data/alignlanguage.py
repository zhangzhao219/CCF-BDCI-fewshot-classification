import os
import json
from paddle import expand
from tqdm import tqdm
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Translate Dataframe')
parser.add_argument('--filefolder', type=str, default='./', help='Data Folder')
parser.add_argument('--ori_train_filename', type=str, required=True, help='src Train FileName')
parser.add_argument('--ori_test_filename', type=str, required=True, help='src Test FileName')
parser.add_argument('--translate_filename', type=str, required=True, help='src Translate FileName')
parser.add_argument('--dst_filename', type=str, required=True, help='dst FileName')
args = parser.parse_args()

# python3 alignlanguage.py --filefolder ./ --ori_train_filename train_en_zh.json --ori_test_filename testA_en_zh.json --translate_filename expand_train_cur_best_en.json --dst_filename expand_train_cur_best_en_zh.json

# python3 alignlanguage.py --filefolder ./ --ori_train_filename train_en.json --ori_test_filename testA_en.json --translate_filename expand_train_cur_best.json --dst_filename expand_train_cur_best_en.json

def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r", encoding='UTF-8') as f:
        reader = f.readlines()
    return [json.loads(line.strip()) for line in reader]

data_ori_test = pd.DataFrame.from_records(read_json(os.path.join(args.filefolder,args.ori_test_filename)))
data_ori_test['label_id'] = -1
data_ori_train = pd.DataFrame.from_records(read_json(os.path.join(args.filefolder,args.ori_train_filename)))
data_ori = pd.concat([data_ori_train, data_ori_test])

data_translate = pd.DataFrame.from_records(read_json(os.path.join(args.filefolder,args.translate_filename)))

for idx, row in tqdm(data_translate.iterrows(), total=data_translate.shape[0]):
    id = row['id']
    en = data_ori.loc[data_ori['id'] == id]
    row['title'] = en['title'].item()
    row['assignee'] = en['assignee'].item()
    row['abstract'] = en['abstract'].item()
    data_translate.iloc[idx] = row

expand_json = data_translate.to_json(orient="records",force_ascii=False)
expand_json = json.loads(expand_json)

with open(os.path.join(os.path.join(args.filefolder,args.dst_filename)), 'w', encoding='utf-8') as f:
    for item in expand_json:
        line = json.dumps(item, ensure_ascii=False)
        f.write(line + '\n')