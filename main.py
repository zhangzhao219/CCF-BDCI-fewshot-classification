import os
import json
import tqdm
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader,Subset,random_split
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold,StratifiedKFold
from transformers.optimization import get_linear_schedule_with_warmup

from model import BertModel,getTokenizer
from dataset import LoadData
from log import config_logging

parser = argparse.ArgumentParser(description='Learning Pytorch NLP')
parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--checkpoint',type=int, default=0, help='Use checkpoint')
parser.add_argument('--data_dir',type=str, required=True, help='Data Location')
parser.add_argument('--epoch', type=int, default=5, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--K', type=int, default=1, help='K-fold')
parser.add_argument('--load', action='store_true', help='load from checkpoint')
parser.add_argument('--load_pt', type=str, help='load from checkpoint')
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--predict', action='store_true', help='Whether to predict')
parser.add_argument('--save', action='store_true', help='Whether to save model')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--warmup',type=float, default=0.1, help='warm up ratio')
args = parser.parse_args()

BERT = 'ernie'
MODEL_PATH = './models/'
DATA_PATH = './data/'
TRAIN_DATA = DATA_PATH + args.data_dir + '/train.json'
TEST_DATA = DATA_PATH + args.data_dir + '/testA.json'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# log
config_logging("log")
logging.info('Log is ready!')

if args.board:
    writer = SummaryWriter()
    logging.info('Tensorboard is ready!')

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    logging.info('GPU:' + ','.join(args.gpu))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info('Seed:' + str(seed))


def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, "r") as f:
        reader = f.readlines()
    return [json.loads(line.strip()) for line in reader]


# () read data and process
def read_data(data_path):
    df = pd.DataFrame.from_records(read_json(data_path))
    df['input_string'] = df.apply(lambda x: f"这份专利的标题为：《{x.title}》，由“{x.assignee}”公司申请，详细说明如下：{x.abstract}",axis=1)
    if len(df.columns) == 5:
        df['label_id'] = 0
    data = df[['id','input_string','label_id']]
    data.columns = ['id','sentence','label']
    # data = pd.read_csv(data_path,names=['id','sentence','label'])
    logging.info('Read data: ' + data_path)
    return data

# # () build vocabulary
# def build_vocab(data):
#     word_list = []
#     # find all words
#     for index,column in data.iterrows():
#         for word in column['sentence'].split():
#             word_list.append(word)
#     # remove duplicate words
#     word_list = sorted(list(set(word_list)))
#     # {word : index}
#     word_dict = {word:index+1 for index,word in enumerate(word_list)}
#     word_dict['<PAD>'] = 0
#     word_dict['<UNK>'] = len(word_dict)
#     id_to_word_dict = {i:j for j,i in word_dict.items()}
#     logging.info('Build Vocab')
#     return word_dict,id_to_word_dict

# evaluate method
def calculateMetrics(label,prediction):
    return f1_score(label,prediction,average='macro')

def train_one_epoch(args,train_loader,model,optimizer,scheduler,criterion,epoch):
    # set train mode
    model.train()
    
    loss = 0

    # define tqdm
    epoch_iterator = tqdm.tqdm(train_loader, desc="Iteration", total=len(train_loader))
    # set description of tqdm
    epoch_iterator.set_description(f'Train-{epoch}')

    for step, (input_id,label) in enumerate(epoch_iterator):
        # print(sentence,label)
        mask = input_id['attention_mask']
        input_id = input_id['input_ids'].squeeze(1)
        if args.gpu:
            model = model.cuda()
            mask = mask.cuda()
            input_id = input_id.cuda()
        output = model(input_id,mask)
        if args.gpu:
            output = output.cpu()
        loss_single = criterion(output, label)
        loss += loss_single.item()

        # renew tqdm
        epoch_iterator.update(1)
        # add description in the end
        epoch_iterator.set_postfix(loss=loss_single.item())

        # backward 
        model.zero_grad() # zero grad
        loss_single.backward()
        optimizer.step()
        scheduler.step()

    return loss / args.batch,eval_one_epoch(args,train_loader,model,epoch)

def eval_one_epoch(args,eval_loader,model,epoch):
    # test
    model.eval()
    epoch_iterator = tqdm.tqdm(eval_loader, desc="Iteration", total=len(eval_loader))
    # set description of tqdm
    epoch_iterator.set_description(f'Eval-{epoch}')

    prob_all = []
    label_all = []
    for step, (input_id,label) in enumerate(epoch_iterator):

        mask = input_id['attention_mask']
        input_id = input_id['input_ids'].squeeze(1)
        if args.gpu:
            model = model.cuda()
            mask = mask.cuda()
            input_id = input_id.cuda()
        with torch.no_grad():
            output = model(input_id,mask)
        if args.gpu:
            output = output.cpu()
        predict = output.argmax(axis=1).numpy().tolist()
        prob_all.extend(predict)
        label_all.extend(label)
        epoch_iterator.update(1)
    metrics = calculateMetrics(label_all,prob_all)
    return metrics


def foldData(kfold,all_dataset,dataset_len,K,index):
    if K >= 2:
        train_index,eval_index = list(kfold.split(all_dataset.texts, all_dataset.labels))[index]
        train_dataset = Subset(all_dataset, train_index)
        eval_dataset = Subset(all_dataset, eval_index)
    else:
        train_dataset, eval_dataset = random_split(all_dataset, [round(dataset_len*0.7), dataset_len-round(dataset_len*0.7)])   
    return train_dataset,eval_dataset
        
# train process
def train(args,data):
    # cross validation
    if args.K >= 2:
        kfold = StratifiedKFold(n_splits=args.K, shuffle=False)
    # store metrics
    best_metrics_list = [0 for i in range(args.K)]
    # cross validation or not (if not, args.K=1 one time)
    for n_fold in range(args.K):
        # build model
        model = BertModel(n_labels=36)
        # use GPU
        if args.gpu:
            model = model.cuda()
            if len(args.gpu) >= 2:
                model= nn.DataParallel(model)
        
        logging.info(f'Start Training {n_fold}!')
        tokenizer = getTokenizer(BERT)
        # data
        all_dataset = LoadData(data,tokenizer)
        # len(data)
        dataset_len = all_dataset.__len__()
        # loss function
        criterion = nn.CrossEntropyLoss()
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        # warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = round(dataset_len / args.batch) * args.epoch, num_training_steps = dataset_len * args.epoch)
        # sign for cross validation or not
        K = n_fold
        if args.K >= 2:
            K += 1
        # restore from checkpoint
        if args.load:
            logging.info(f'Load checkpoint_{args.load_pt}_{K}_epoch.pt')
            checkpoint = torch.load(f'{MODEL_PATH}checkpoint_{args.load_pt}_{K}_epoch.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # train test split (kfold or not)
        train_dataset,eval_dataset = foldData(kfold,all_dataset,dataset_len,args.K,n_fold)
        # loops
        for epoch in range(args.epoch):
            # dataset loader
            train_loader = DataLoader(train_dataset,batch_size=args.batch,shuffle=True,drop_last=False)
            eval_loader = DataLoader(eval_dataset,batch_size=args.batch*8,shuffle=False,drop_last=False)
            # train and evaluate train dataset
            loss,metrics = train_one_epoch(args,train_loader,model,optimizer,scheduler,criterion,epoch+1)
            logging.info(f'Train Epoch = {epoch+1} Loss:{loss:{.6}} Metrics:{metrics:{.6}}')
            # evaluate eval dataset
            metrics = eval_one_epoch(args,eval_loader,model,epoch+1)
            logging.info(f'Eval Epoch = {epoch+1} Metrics:{metrics:{.6}}')
            # save model
            if args.checkpoint != 0 and epoch % args.checkpoint == 0:
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss}, MODEL_PATH + 'checkpoint_{}_{}_epoch.pt'.format(epoch,K))
                logging.info(f'checkpoint_{epoch}_{K}_epoch.pt Saved!')
            # save better model
            if args.save and metrics > best_metrics_list[n_fold]:
                torch.save(model.state_dict(), MODEL_PATH + 'best_{}.pt'.format(K))
                logging.info(f'Test metrics:{metrics:{.6}} > max_accuracy! Saved!')
                best_metrics_list[n_fold] = metrics
            # tensorboard
            if args.board:
                writer.add_scalar('loss', loss, epoch)
                writer.add_scalar('metrics', metrics, epoch)
        # clear gpu parameters
        if args.gpu:
            torch.cuda.empty_cache()

# use any model to evaluate labeled data
def test(args,data,mode):
    logging.info('Start evaluate!')

    # test data
    test_dataset = LoadData(data,getTokenizer(BERT))
    test_loader = DataLoader(test_dataset,batch_size=args.batch,shuffle=False,drop_last=False)

    labellist = []
    predict_result = np.empty((args.K,test_dataset.__len__(),36))
    
    for n_fold in range(args.K):
        K = n_fold
        if args.K >= 2:
            K += 1
        # build model
        model = BertModel(n_labels=36)
        # use GPU
        if args.gpu:
            model = model.cuda()
            if len(args.gpu) >= 2:
                model= nn.DataParallel(model)
        # load best model
        model.load_state_dict(torch.load(MODEL_PATH + 'best_{}.pt'.format(K)))
        logging.info(f'best_{K}.pt Loaded!')
        # set eval mode
        model.eval()

        epoch_iterator = tqdm.tqdm(test_loader, desc="Iteration", total=len(test_loader))
        # set description of tqdm
        epoch_iterator.set_description('Test')

        for step, (input_id,label) in enumerate(epoch_iterator):
            mask = input_id['attention_mask']
            input_id = input_id['input_ids'].squeeze(1)
            if args.gpu:
                model = model.cuda()
                mask = mask.cuda()
                input_id = input_id.cuda()
            with torch.no_grad():
                output = model(input_id,mask)
            if args.gpu:
                output = output.cpu()
            output = output.numpy()
            output_shape = output.shape
            predict_result[n_fold][step*args.batch:step*args.batch+output_shape[0]] = output

            if n_fold == 0:
                labellist.extend(label)

            epoch_iterator.update(1)
        if mode == 0:
            # calculate single model metrics
            metrics = calculateMetrics(labellist,predict_result[n_fold].argmax(axis=1).tolist())
            logging.info(f'Metrics for best_{K}.pt : {metrics}')
    if mode == 1:
        return predict_result
    # calculate final metrics
    metrics = calculateMetrics(labellist,predict_result.mean(axis=0).argmax(axis=1).tolist())
    logging.info(f'Metrics for all model : {metrics}')

# predict unlabeled data
def predict(args,data):
    predict_result = test(args,data,1)
    predict_result = predict_result.mean(axis=0).argmax(axis=1)
    data['label'] = predict_result
    data[['id','label']].to_csv(DATA_PATH + args.data_dir +'/result.csv',index=None)
    logging.info('Predict Finished!')

if __name__ == '__main__':
    # set seed
    set_seed(args.seed)
    # print(args.batch, args.epoch, args.eval, args.gpu, args.seed, args.train)

    # read data
    train_data = read_data(TRAIN_DATA)
    test_data = read_data(TEST_DATA)
    
    # build vocab
    # word_dict,id_to_word = build_vocab(pd.concat([train_data, test_data], axis=0))
    # word_dict = build_vocab(train_data)
    # vocab_size = len(word_dict)
    # print(word_dict)

    # config
    # n_step = 3 # number of cells(= number of Step)
    # embedding_dim = 768 # embedding size
    # n_hidden = 768  # number of hidden units in one cell
    # num_classes = 36  # 0 or 1
    
    if args.train:
        train(args,train_data)
    if args.test:
        test(args,train_data,0)
    if args.predict:
        predict(args,test_data)