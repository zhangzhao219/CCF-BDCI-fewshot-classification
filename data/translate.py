import os
import json
import time
import shutil
import argparse
from tqdm import tqdm

import pandas as pd
from googletrans import Translator

parser = argparse.ArgumentParser(description='Translate Dataframe')
parser.add_argument('--filefolder', type=str, default='./', help='Translate FileFolder')
parser.add_argument('--filename', type=str, required=True, help='src Translate FileName')
parser.add_argument('--src_language', type=str, default='zh-cn', help='Original language')
parser.add_argument('--dst_language', type=str, default='en', help='Translate language')
parser.add_argument('--begin_line', type=int, default=0, help='Where to start')

args = parser.parse_args()

# 使用前copy文件并命好名，后续是直接在这个文件上进行修改的
# python3 trans.py --filefolder ./ --filename train_en_zh --src_language en --dst_language zh-cn --begin_line 0

translator = Translator(service_urls=[
    'translate.google.com'
])

def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r", encoding='UTF-8') as f:
        reader = f.readlines()
    return [json.loads(line.strip()) for line in reader]

data = pd.DataFrame.from_records(read_json(os.path.join(args.filefolder,args.filename)+'.json'))

print(os.path.join(args.filefolder,args.filename)+'.json')

for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
    if idx >= args.begin_line:
        while True:
            try:
                row['title'] = translator.translate(row['title'],src=args.src_language,dest=args.dst_language).text
                row['assignee'] = translator.translate(row['assignee'],src=args.src_language,dest=args.dst_language).text
                row['abstract'] = translator.translate(row['abstract'],src=args.src_language,dest=args.dst_language).text
                data.iloc[idx] = row
                break
            except:
                print(idx," Retrying")
                time.sleep(1)
            finally:
                file = data.to_json(orient="records",force_ascii=False)
                file = json.loads(file)

                with open(os.path.join(args.filefolder,args.filename)+'.json', 'w', encoding='utf-8') as f:
                    for item in file:
                        line = json.dumps(item, ensure_ascii=False)
                        f.write(line + '\n')

                shutil.copy(os.path.join(args.filefolder,args.filename)+'.json',os.path.join(args.filefolder,args.filename)+'_copy.json')