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
from torch.utils.data import DataLoader,Subset,random_split

from tensorboardX import SummaryWriter

from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,StratifiedKFold

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from model import BertModel,RDrop,EMA,PGD,FGM,getTokenizer
from dataset import LoadData
from log import config_logging

parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--predict', action='store_true', help='Whether to predict')

parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--epoch', type=int, default=5, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')

parser.add_argument('--data_folder_dir', type=str, required=True, help='Data Folder Location')
parser.add_argument('--data_file', type=str, help='Data Filename')

parser.add_argument('--checkpoint', type=int, default=0, help='Use checkpoint')
parser.add_argument('--load', action='store_true', help='load from checkpoint')
parser.add_argument('--load_pt', type=str, help='load from checkpoint')
parser.add_argument('--save', action='store_true', help='Whether to save model')

parser.add_argument('--bert', type=str, required=True, help='Choose Bert')
parser.add_argument('--K', type=int, default=1, help='K-fold')
parser.add_argument('--warmup', type=float, default=0.1, help='warm up ratio')
parser.add_argument('--rdrop', type=float, default=0.0, help='RDrop kl_weight')
parser.add_argument('--ema', type=float, default=0.0, help='EMA decay')
parser.add_argument('--fgm', type=float, default=0.0, help='FGM epsilon')
parser.add_argument('--pgd', type=int, default=0, help='PGD K')

args = parser.parse_args()

TIMESTAMP = args.datetime

# log
config_logging("log_" + TIMESTAMP)
logging.info('Log is ready!')
logging.info(args)

if args.board:
    writer = SummaryWriter('runs/' + TIMESTAMP)
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
    P = round(precision_score(label,prediction,average='macro',zero_division=0),6)
    R = round(recall_score(label,prediction,average='macro',zero_division=0),6)
    F1 = round(f1_score(label,prediction,average='macro',zero_division=0),6)
    return {'F1score':F1,'Precision':P,'Recall':R}

# def train_one_epoch(args,train_loader,model,optimizer,scheduler,criterion,epoch):
def train_one_epoch(args,train_loader,model,optimizer,criterion,epoch,ema):
    # set train mode
    model.train()

    if args.fgm != 0.0:
        fgm = FGM(model,args.fgm)
    if args.pgd != 0:
        pgd = PGD(model)
    
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
            label = label.cuda()
        output = model(input_id, mask)
        if args.rdrop != 0.0:
            output_rdrop = model(input_id, mask)
            loss_single = criterion(output, output_rdrop, label, args.rdrop)
        else:
            loss_single = criterion(output, label)
        loss += loss_single.item()
        # backward 
        loss_single.backward()

        # FGM attack
        if args.fgm != 0.0:
            fgm.attack() # attack on embedding
            output_adv = model(input_id, mask)
            loss_adv = criterion(output_adv, label)
            loss_adv.backward()
            fgm.restore() # restore embedding

        # PGD attack
        if args.pgd != 0:
            pgd.backup_grad()
            for t in range(args.pgd):
                pgd.attack(is_first_attack=(t==0)) # attack on embedding,and backup param.data when first attack
                if t != args.pgd-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                output_adv = model(input_id, mask)
                loss_adv = criterion(output_adv, label)
                loss_adv.backward()
            pgd.restore() # 恢复embedding参数

        optimizer.step()
        # scheduler.step()
        if args.ema != 0.0:
            ema.update()

        model.zero_grad() # zero grad

        # renew tqdm
        epoch_iterator.update(1)
        # add description in the end
        epoch_iterator.set_postfix(loss=loss_single.item())

    return loss / args.batch, eval_one_epoch(args, train_loader, model, epoch)

def eval_one_epoch(args, eval_loader, model, epoch):
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
        train_dataset, eval_dataset = random_split(all_dataset, [round(dataset_len*0.8), dataset_len-round(dataset_len*0.8)])   
    return train_dataset,eval_dataset
        
# train process
def train(args,data):
    # cross validation
    if args.K >= 2:
        kfold = StratifiedKFold(n_splits=args.K, shuffle=False)
    else:
        kfold = None
    # store metrics
    best_metrics_list = [0 for i in range(args.K)]
    # cross validation or not (if not, args.K=1 one time)
    for n_fold in range(args.K):
        # build model
        model = BertModel(bert=args.bert,n_labels=36)
        # use GPU
        if args.gpu:
            model = model.cuda()
            if len(args.gpu) >= 2:
                model= nn.DataParallel(model)
        if args.ema != 0.0:
            ema = EMA(model, args.ema)
            ema.register()
        else:
            ema = None
        # sign for cross validation or not
        K = n_fold
        if args.K >= 2:
            K += 1
        logging.info(f'Start Training {K}!')
        tokenizer = getTokenizer(args.bert)
        # data
        all_dataset = LoadData(data,tokenizer)
        # len(data)
        dataset_len = all_dataset.__len__()
        # loss function
        # whether to use RDrop
        if args.rdrop != 0.0:
            criterion = RDrop()
        else:
            criterion = nn.CrossEntropyLoss()
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer = AdamW(model.parameters(), lr=args.lr,correct_bias=True)
        # warmup
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = round(dataset_len / args.batch) * args.epoch, num_training_steps = dataset_len * args.epoch)
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
            # loss,metrics = train_one_epoch(args,train_loader,model,optimizer,scheduler,criterion,epoch+1)
            loss,metrics_train = train_one_epoch(args, train_loader, model, optimizer, criterion, epoch+1, ema)
            logging.info(f'Train Epoch = {epoch+1} Loss:{loss:{.6}} Metrics:{metrics_train}')
            if args.ema != 0.0:
                ema.apply_shadow()
            # evaluate eval dataset
            metrics = eval_one_epoch(args,eval_loader,model,epoch+1)
            logging.info(f'Eval Epoch = {epoch+1} Metrics:{metrics}')
            # save model
            if args.checkpoint != 0 and epoch % args.checkpoint == 0:
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss}, MODEL_PATH + 'checkpoint_{}_{}_epoch.pt'.format(epoch,K))
                logging.info(f'checkpoint_{epoch}_{K}_epoch.pt Saved!')
            # save better model
            if args.save and metrics[list(metrics.keys())[0]] > best_metrics_list[n_fold]:
                torch.save(model.state_dict(), MODEL_PATH + 'best_{}.pt'.format(K))
                logging.info(f'Test metrics:{metrics[list(metrics.keys())[0]]} > max_metric! Saved!')
                best_metrics_list[n_fold] = metrics[list(metrics.keys())[0]]
            # tensorboard
            if args.board:
                writer.add_scalar(f'K_{K}/Loss', loss, epoch+1)
                for i in list(metrics.keys()):
                    writer.add_scalars(f'K_{K}/{i}', {'Train_'+i:metrics_train[i],'Eval_'+i:metrics[i]}, epoch+1)
            # if args.ema != 0.0:
            #     ema.restore()
        # clear gpu parameters
        if args.gpu:
            torch.cuda.empty_cache()

# use any model to evaluate labeled data
def test(args,data,mode):
    logging.info('Start evaluate!')

    # test data
    test_dataset = LoadData(data,getTokenizer(args.bert))
    test_loader = DataLoader(test_dataset,batch_size=args.batch,shuffle=False,drop_last=False)

    labellist = []
    predict_result = np.empty((args.K,test_dataset.__len__(),36))
    
    for n_fold in range(args.K):
        K = n_fold
        if args.K >= 2:
            K += 1
        # build model
        model = BertModel(bert=args.bert,n_labels=36)
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
            output = nn.Softmax(dim=-1)(output).numpy()
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
    data[['id','label']].to_csv('result_'+TIMESTAMP+'.csv',index=None)
    logging.info('Predict Finished!')

if __name__ == '__main__':
    # set seed
    set_seed(args.seed)
    # print(args.batch, args.epoch, args.eval, args.gpu, args.seed, args.train)

    MODEL_PATH = './models/' + TIMESTAMP + '/'
    DATA_PATH = './data/'
    DATA_PATH = os.path.join(DATA_PATH,args.data_folder_dir,args.data_file)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # read data
    DATA = read_data(DATA_PATH)
    
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
        train(args,DATA)
    if args.test:
        test(args,DATA,0)
    if args.predict:
        predict(args,DATA)