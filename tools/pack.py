import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Learning Pytorch NLP')

parser.add_argument('--datetime',type=str, required=True, help='Time Stamp')
parser.add_argument('--score',type=float, required=True, help='Model score')

args = parser.parse_args()

savepath = './goodmodels/'+args.datetime+'_'+str(args.score)

if not os.path.exists(savepath):
    os.mkdir(savepath)

shutil.copytree('./runs/'+args.datetime, savepath+'/runs')
shutil.copytree('./models/'+args.datetime, savepath+'/models')
shutil.copy('log_'+args.datetime, savepath)
shutil.copy('result_'+args.datetime+'.csv', savepath)