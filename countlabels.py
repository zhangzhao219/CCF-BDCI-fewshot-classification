import sys
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--data_file', type=str, required=True, help='Data Filename')

args = parser.parse_args()


def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r") as f:
        reader = f.readlines()
    return [json.loads(line.strip()) for line in reader]

if __name__ == '__main__':
    if args.data_file.split('.')[-1] == 'json':
        df = pd.DataFrame.from_records(read_json(args.data_file))
        name = 'label_id'
    elif args.data_file.split('.')[-1] == 'csv':
        df = pd.read_csv(args.data_file)
        name = 'label'
    else:
        print('No support format!')
        sys.exit()

    label_dict = dict(df[name].value_counts())
    dict = sorted(label_dict.items(), key=lambda d: d[0], reverse=False)
    print(dict)
    print([i[1] for i in list(dict)])
    print(sum([i[1] for i in list(dict)]))
    