import os
import ast
import json
import tqdm
import torch
import logging
import argparse
import importlib

import numpy as np
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LoadData
from log import config_logging

parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--batch', type=int, default=128, help='Define the batch size')
parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')

parser.add_argument('--data_file', help='Data Filename')

parser.add_argument('--label', type=int, default=36, help='label num')

parser.add_argument('--model_dict', type=ast.literal_eval)
parser.add_argument('--model_weight', type=ast.literal_eval)

parser.add_argument('--score', action='store_true', help='whether to output score')
parser.add_argument('--single', action='store_true', help='whether to output single result')
parser.add_argument('--softmax', action='store_true', help='whether to output softmax score')

args = parser.parse_args()

MODEL_DICT = args.model_dict
MODEL_WEIGHT = args.model_weight
TIMESTAMP = args.datetime

# log
config_logging("../../user_data/log/log_ensemble_" + TIMESTAMP)
logging.info('Ensemble Log is Ready!')
logging.info(args)

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    logging.info('GPU:' + ','.join(args.gpu))

def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r") as f:
        reader = f.readlines()
    return [json.loads(line.strip()) for line in reader]

def read_data(data_path, language):
    df = pd.DataFrame.from_records(read_json(data_path))
    if language == 'en':
        df['input_string'] = df.apply(lambda x: f"The name of the patent is {x.title}, applied for by {x.assignee}, and the details are as follows: {x.abstract}",axis=1)
    elif language == 'zh':
        df['input_string'] = df.apply(lambda x: f"这份专利的标题为：《{x.title}》，由“{x.assignee}”公司申请，详细说明如下：{x.abstract}",axis=1)
    else:
        print('No language!')
    if len(df.columns) == 5:
        df['label_id'] = 0
    data = df[['id','input_string','label_id']]
    data.columns = ['id','sentence','label']
    logging.info('Read data: ' + data_path)
    return data

# predict single model
def predict_single(args, data, model_config, index):

    logging.info(f'Start evaluate {index}!')

    model_structure = importlib.import_module(model_config['model'])
    PretrainedModel = model_structure.PretrainedModel
    getTokenizer = model_structure.getTokenizer

    # test data
    test_dataset = LoadData(data, getTokenizer(model_config['bert']))
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, drop_last=False)

    model = PretrainedModel(model_config['bert'], args.label, model_config['feature_layers'], model_config['dropout'])
    # use GPU
    if args.gpu:
        model = model.cuda()
        model = nn.DataParallel(model)

    # load best model
    model.load_state_dict(torch.load(model_config['name']), not model_config['swa'])
    logging.info(f"{model_config['name']} Loaded!")

    # set eval mode
    model.eval()

    predict_result = np.empty((test_dataset.__len__(), args.label))

    epoch_iterator = tqdm.tqdm(test_loader, desc="Iteration", total=len(test_loader))
    # set description of tqdm
    epoch_iterator.set_description('Test')

    for step, ((input_ids, token_type_ids, attention_mask), label) in enumerate(epoch_iterator):

        input_ids = input_ids.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        if args.gpu:
            model = model.cuda()
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            
        with torch.no_grad():
            output = model(input_ids, token_type_ids, attention_mask)
        if args.gpu:
            output = output.cpu()
        output = nn.Softmax(dim=-1)(output).numpy()
        output_shape = output.shape
        predict_result[step*args.batch:step*args.batch+output_shape[0]] = output

        epoch_iterator.update(1)

    predict_score_single = predict_result.max(axis=1)
    predict_result_single = predict_result.argmax(axis=1)

    output_list = ['id','label']
    if args.softmax:
        for i in range(args.label):
            data[str(i)] = predict_result[:,i]
            output_list.append(str(i))

    if args.single:
        data['label'] = predict_result_single
        if not args.score:
            data[output_list].to_csv('result_'+TIMESTAMP+'_'+str(index)+'.csv',index=None)
        else:
            output_list.append('score')
            data['score'] = predict_score_single
            data[output_list].to_csv('result_score_'+TIMESTAMP+'_'+str(index)+'.csv',index=None)

    logging.info(f'Predict {index} Finished!')
    return predict_result

if __name__ == '__main__':

    predict_result_list = []

    for (index, model_parameters) in enumerate(MODEL_DICT):
        DATA = read_data(args.data_file, model_parameters['language'])
        predict_result_single = predict_single(args, DATA, model_parameters, index+1)
        predict_result_list.append(predict_result_single)

    
    
    model_weight = np.array(args.model_weight) / sum(args.model_weight)
    predict_result = np.sum(np.array(predict_result_list) * model_weight.reshape(-1,1,1),axis=0)

    predict_result_single = predict_result.argmax(axis=1)

    output_data = DATA

    output_list = ['id','label']
    if args.softmax:
        for i in range(args.label):
            output_data[str(i)] = predict_result[:,i]
            output_list.append(str(i))

    output_data['label'] = predict_result_single
    if not args.score:
        output_data[output_list].to_csv('../../prediction_result/finalB.csv',index=None)
    else:
        output_list.append('score')
        predict_score_single = predict_result.max(axis=1)
        output_data['score'] = predict_score_single
        output_data[output_list].to_csv('result_score_'+TIMESTAMP+'_all.csv',index=None)

    logging.info(f'Predict Finished!')