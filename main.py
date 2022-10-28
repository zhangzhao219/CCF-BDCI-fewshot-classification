import os
import json
import tqdm
import torch
import random
import logging
import argparse

import numpy as np
import pandas as pd
from collections import deque

import torch.nn as nn
from torch.utils.data import DataLoader,Subset

from tensorboardX import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from trick import RDrop,EMA,PGD,FGM
from dataset import LoadData
from log import config_logging

parser = argparse.ArgumentParser(description='Pytorch NLP')

parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--predict', action='store_true', help='Whether to predict')
parser.add_argument('--predict_with_score', action='store_true', default=False, help='Whether to predict')

parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--epoch', type=int, default=50, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')
parser.add_argument('--early_stop',type=int, default=10, help='Early Stop Epoch')

parser.add_argument('--data_folder_dir', type=str, required=True, help='Data Folder Location')
parser.add_argument('--data_file', type=str, help='Data Filename')
parser.add_argument('--label', type=int, default=36, help='label num')

parser.add_argument('--checkpoint', type=int, default=0, help='Use checkpoint')
parser.add_argument('--load', action='store_true', help='load from checkpoint')
parser.add_argument('--load_pt', type=str, help='load from checkpoint')
parser.add_argument('--save', action='store_true', help='Whether to save model')

parser.add_argument('--bert', type=str, required=True, help='Choose Bert')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout ratio')
parser.add_argument('--feature_layer', type=int, default=4, help='feature layers num')
parser.add_argument('--freeze', type=int, default=0, help='freeze bert parameters')

parser.add_argument('--ema', type=float, default=0.0, help='EMA decay')
parser.add_argument('--swa', action='store_true', help='swa ensemble')
parser.add_argument('--fgm', action='store_true', help='FGM attack')
parser.add_argument('--K', type=int, default=1, help='K-fold')
parser.add_argument('--pgd', type=int, default=0, help='PGD K')
parser.add_argument('--rdrop', type=float, default=0.0, help='RDrop kl_weight')
parser.add_argument('--split_test_ratio', type=float, default=0.2, help='if no Kfold, split test ratio')
parser.add_argument('--warmup', type=float, default=0.0, help='warm up ratio')

args = parser.parse_args()

TIMESTAMP = args.datetime

if args.bert.split('/')[-1] == "roberta-base":
    from roberta_model import PretrainedModel,getTokenizer
elif args.bert.split('/')[-1] == "xlnet-base-cased":
    from xlnet_model import PretrainedModel,getTokenizer
else:
    from auto_bert_model import PretrainedModel,getTokenizer

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
    # df['input_string'] = df.apply(lambda x: f"这份专利的标题为：《{x.title}》，由“{x.assignee}”公司申请，详细说明如下：{x.abstract}",axis=1)
    df['input_string'] = df.apply(lambda x: f"The name of the patent is {x.title}, applied for by {x.assignee}, and the details are as follows: {x.abstract}",axis=1)
    if len(df.columns) == 5:
        df['label_id'] = 0
    data = df[['id','input_string','label_id']]
    data.columns = ['id','sentence','label']
    # data = pd.read_csv(data_path,names=['id','sentence','label'])
    logging.info('Read data: ' + data_path)
    return data

# evaluate method
def calculateMetrics(label,prediction):
    P = round(precision_score(label,prediction,average='macro',zero_division=0),6)
    R = round(recall_score(label,prediction,average='macro',zero_division=0),6)
    F1 = round(f1_score(label,prediction,average='macro',zero_division=0),6)
    return {'F1score':F1,'Precision':P,'Recall':R}

def train_one_epoch(args, train_loader, model, optimizer, scheduler, criterion, epoch, ema):

    # define tqdm
    epoch_iterator = tqdm.tqdm(train_loader, desc="Iteration", total=len(train_loader))
    # set description of tqdm
    epoch_iterator.set_description(f'Train-{epoch}')

    # set train mode
    model.train()

    # fgm
    if args.fgm:
        fgm = FGM(model)

    # pgd
    if args.pgd != 0:
        pgd = PGD(model)
    
    loss = 0

    for step, ((input_ids, token_type_ids, attention_mask), label) in enumerate(epoch_iterator):

        input_ids = input_ids.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        if args.gpu:
            model = model.cuda()
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            label = label.cuda()

        output = model(input_ids, token_type_ids, attention_mask)

        # rdrop
        if args.rdrop != 0.0:
            output_rdrop = model(input_ids, token_type_ids, attention_mask)
            loss_single = criterion(output, output_rdrop, label, args.rdrop)
        else:
            loss_single = criterion(output, label)

        loss += loss_single.item()

        # backward 
        loss_single.backward()

        # FGM attack
        if args.fgm:
            fgm.attack() # attack on embedding
            output_adv = model(input_ids,token_type_ids,attention_mask)
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
                output_adv = model(input_ids, token_type_ids, attention_mask)
                loss_adv = criterion(output_adv, label)
                loss_adv.backward()
            pgd.restore() # 恢复embedding参数

        optimizer.step()

        if args.warmup != 0.0:
            scheduler.step()

        # 打印学习率
        # print(optimizer.state_dict()['param_groups'][0]['lr'])

        if args.ema != 0.0:
            ema.update(warmup_if = epoch < args.epoch / 4)

        model.zero_grad() # zero grad

        # renew tqdm
        epoch_iterator.update(1)
        # add description in the end
        epoch_iterator.set_postfix(loss=loss_single.item())

    return loss / args.batch, eval_one_epoch(args, train_loader, model, epoch)

def eval_one_epoch(args, eval_loader, model, epoch):

    epoch_iterator = tqdm.tqdm(eval_loader, desc="Iteration", total=len(eval_loader))
    # set description of tqdm
    epoch_iterator.set_description(f'Eval-{epoch}')

    # test
    model.eval()

    prob_all = []
    label_all = []
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
        predict = output.argmax(axis=1).numpy().tolist()
        prob_all.extend(predict)
        label_all.extend(label)

        epoch_iterator.update(1)
    metrics = calculateMetrics(label_all,prob_all)
    return metrics


def foldData(kfold,all_dataset,K,index,ratio):
    if K >= 2:
        train_index,eval_index = list(kfold.split(all_dataset, all_dataset.labels))[index]
        train_dataset = Subset(all_dataset, train_index)
        eval_dataset = Subset(all_dataset, eval_index)
    else:
        train_dataset, eval_dataset, a, b = train_test_split(all_dataset, all_dataset.labels, test_size=ratio, random_state=args.seed, stratify = all_dataset.labels)
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
        model = PretrainedModel(args.bert, args.label, args.feature_layer, args.dropout)

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

        # 冻结除了feature_layer外其他的参数
        if args.freeze != 0:
            # for name, param in model.named_parameters():
            #     print(name,param.size())
            unfreeze_layers = ['layer.'+str(i) for i in range(args.freeze,12)]
            unfreeze_layers.extend(['bert.pooler','linear.']) # 注意不能把自己加上去的层也锁了！！！
            # print(unfreeze_layers)
            for name, param in model.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break
            # # 验证一下
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name,param.size())

        # 由于在bert官方的代码中对于bias项、LayerNorm.bias、LayerNorm.weight项是免于正则化的。因此经常在bert的训练中会采用与bert原训练方式一致的做法
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,correct_bias=True)
        scheduler = optimizer
        if args.swa:
            swa_model = AveragedModel(model).to('cuda')
            swa_scheduler = SWALR(optimizer, swa_lr=args.lr)
        # warmup
        if args.warmup != 0.0:
            if args.K == 1:
                num_train_optimization_steps = dataset_len / args.batch * (1-args.split_test_ratio) * args.epoch
            else:
                num_train_optimization_steps = dataset_len / args.batch * (args.K-1) / args.K * args.epoch
            scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps*args.warmup), num_train_optimization_steps)

        # restore from checkpoint
        if args.load:
            logging.info(f'Load checkpoint_{args.load_pt}_{K}_epoch.pt')
            checkpoint = torch.load(f'{MODEL_PATH}checkpoint_{args.load_pt}_{K}_epoch.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # train test split (kfold or not)
        train_dataset,eval_dataset = foldData(kfold,all_dataset,args.K,n_fold,args.split_test_ratio)
        # early stop
        early_stop_sign = deque(maxlen=args.early_stop)
        # loops
        for epoch in range(args.epoch):
            # dataset loader
            train_loader = DataLoader(train_dataset,batch_size=args.batch,shuffle=True,drop_last=False)
            eval_loader = DataLoader(eval_dataset,batch_size=args.batch*8,shuffle=False,drop_last=False)
            # train and evaluate train dataset
            loss,metrics_train = train_one_epoch(args, train_loader, model, optimizer, scheduler, criterion, epoch+1, ema)
            logging.info(f'Train Epoch = {epoch+1} Loss:{loss:{.6}} Metrics:{metrics_train}')
            if args.swa:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            if args.ema != 0.0:
                ema.apply_shadow()
            # evaluate eval dataset
            if args.swa:
                metrics = eval_one_epoch(args,eval_loader,swa_model,epoch+1)
            else:
                metrics = eval_one_epoch(args,eval_loader,model,epoch+1)
            logging.info(f'Eval Epoch = {epoch+1} Metrics:{metrics}')
            main_metric = metrics[list(metrics.keys())[0]]
            # save model
            if args.checkpoint != 0 and epoch % args.checkpoint == 0:
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss}, MODEL_PATH + 'checkpoint_{}_{}_epoch.pt'.format(epoch,K))
                logging.info(f'checkpoint_{epoch}_{K}_epoch.pt Saved!')
            # save better model and early_stop
            if main_metric > best_metrics_list[n_fold]:
                logging.info(f'Test metrics:{main_metric} > max_metric!')
                best_metrics_list[n_fold] = main_metric
                early_stop_sign.append(0)
                if args.save:
                    if args.swa:
                        torch.optim.swa_utils.update_bn(train_loader, swa_model, device='cuda')
                        torch.save(swa_model.state_dict(), MODEL_PATH + 'best_{}.pt'.format(K))
                    else:
                        torch.save(model.state_dict(), MODEL_PATH + 'best_{}.pt'.format(K))
                    logging.info(f'Best Model Saved!')
            else:
                early_stop_sign.append(1)
                if sum(early_stop_sign) == args.early_stop:
                    logging.info(f'The Effect of last {args.early_stop} epochs has not improved! Early Stop!')
                    logging.info(f'Best Metric: {best_metrics_list[n_fold]}')
                    break
            # tensorboard
            if args.board:
                if args.warmup != 0.0:
                    writer.add_scalar(f'K_{K}/Learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)
                writer.add_scalar(f'K_{K}/Loss', loss, epoch+1)
                for i in list(metrics.keys()):
                    writer.add_scalars(f'K_{K}/{i}', {'Train_'+i:metrics_train[i],'Eval_'+i:metrics[i]}, epoch+1)
            if args.ema != 0.0:
                ema.restore()
        # clear gpu parameters
        if args.swa:
            torch.save(swa_model.state_dict(), MODEL_PATH + 'last_{}.pt'.format(K))
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
        model = model = PretrainedModel(args.bert, args.label, args.feature_layer, args.dropout)
        # use GPU
        if args.gpu:
            model = model.cuda()
            if len(args.gpu) >= 2 or args.predict or args.predict_with_score:
                model= nn.DataParallel(model)
        # load best model
        model.load_state_dict(torch.load(MODEL_PATH + 'best_{}.pt'.format(K)))
        logging.info(f'best_{K}.pt Loaded!')
        # set eval mode
        model.eval()

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
            predict_result[n_fold][step*args.batch:step*args.batch+output_shape[0]] = output

            if n_fold == 0:
                labellist.extend(label)

            epoch_iterator.update(1)
        if mode == 0:
            # calculate single model metrics
            metrics = calculateMetrics(labellist, predict_result[n_fold].argmax(axis=1).tolist())
            logging.info(f'Metrics for best_{K}.pt : {metrics}')

    if mode == 1:
        return predict_result
    # calculate final metrics
    metrics = calculateMetrics(labellist, predict_result.mean(axis=0).argmax(axis=1).tolist())
    logging.info(f'Metrics for all model : {metrics}')

# predict unlabeled data
def predict(args,data):
    predict_result = test(args,data,1)
    predict_score = predict_result.mean(axis=0).max(axis=1)
    predict_result = predict_result.mean(axis=0).argmax(axis=1)
    if not args.predict_with_score:
        data['label'] = predict_result
        data[['id','label']].to_csv('result_'+TIMESTAMP+'.csv',index=None)
    else:
        data['label'] = predict_result
        data['score'] = predict_score
        data[['id','label','score']].to_csv('result_score_'+TIMESTAMP+'.csv',index=None)
    logging.info('Predict Finished!')

if __name__ == '__main__':

    # set seed
    set_seed(args.seed)

    MODEL_PATH = './models/' + TIMESTAMP + '/'
    DATA_PATH = './data/'
    DATA_PATH = os.path.join(DATA_PATH,args.data_folder_dir,args.data_file)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # read data
    DATA = read_data(DATA_PATH)
    
    if args.train:
        train(args,DATA)
    if args.test:
        test(args,DATA,0)
    if args.predict or args.predict_with_score:
        predict(args,DATA)
