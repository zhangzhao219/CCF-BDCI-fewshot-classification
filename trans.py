import json
import argparse
from tqdm import tqdm

import pandas as pd
from googletrans import Translator

parser = argparse.ArgumentParser(description='Translate Dataframe')
parser.add_argument('--filename', type=str, required=True, help='Translate FileName')
parser.add_argument('--src_language', type=str, default='zh-cn', help='Original language')
parser.add_argument('--dst_language', type=str, default='en', help='Translate language')
parser.add_argument('--begin_line', type=int, default=0, help='Where to start')

# proxychains -q -f ~/proxychains4.conf python3 trans.py --filename ./data/fewshot/train.json --begin_line 0

args = parser.parse_args()

translator = Translator(service_urls=[
    'translate.google.com'
])

def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r", encoding='UTF-8') as f:
        reader = f.readlines()
    return [json.loads(line.strip()) for line in reader]

data = pd.DataFrame.from_records(read_json(args.filename))

try:
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        if idx >= args.begin_line:
            row['title'] = translator.translate(row['title']).text
            row['assignee'] = translator.translate(row['assignee']).text
            row['abstract'] = translator.translate(row['abstract']).text
            data.iloc[idx] = row
finally:
    file = data.to_json(orient="records",force_ascii=False)
    file = json.loads(file)

    with open(args.filename, 'w', encoding='utf-8') as f:
        for item in file:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')