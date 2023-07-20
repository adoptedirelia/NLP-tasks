import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import config
import argparse
from dataset import MovieData
from model import MyBert
import time 

def get_parser():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification')
    parser.add_argument('--config', type=str, default='./config.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def data_prepare(args):
    train_data = MovieData(0,maxlen=args.maxlen)
    test_data = MovieData(1,maxlen=args.maxlen)

    train_iter = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
    test_iter = DataLoader(test_data,batch_size=args.batch_size)

    return train_iter,test_iter

def train(train_iter,args):
    net = MyBert(args.maxlen).cuda()
    print(net)
    net = nn.DataParallel(net)
    net.train()
    #myloss = nn.CrossEntropyLoss()
    myloss = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(),args.lr)
    
    for epoch in range(args.epoch):
        loss_sum,n,acc_sum,c = 0.0,0,0.0,0
        with tqdm(train_iter) as t:
            begin = time.time()
            for X,y in t:

                y = y.cuda().float()
                #print(X.shape,y.shape)
                t.set_description(f"epoch {epoch+1}")
                optimizer.zero_grad()

                # forward + backward + optimize
                y_pre = net(X[0].cuda(),X[1].cuda())
                
                #print(y_pre)
                loss = myloss(y_pre,y)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                #acc_sum += (y_pre.argmax(dim=1)==y).sum().item()
                pre = y_pre>=torch.tensor([0.5]).cuda()
                corr = pre == y
                acc_sum += corr.sum().item()

                n+=1
                c+=y.shape[0]
            loss = loss_sum/n
            acc = acc_sum/c
            end = time.time()
            print(f'epoch {epoch+1}: loss: {loss:.4f} accuracy: {acc:.4f} time consume: {(end-begin):.4f}')
    
    return net

def eval(model,test_iter,args):

    
    #myloss = nn.CrossEntropyLoss()
    myloss = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        loss_sum,n,acc_sum,c = 0.0,0,0.0,0
        with tqdm(test_iter) as t:
            for X,y in t:
                y = y.cuda().float()
                y_pre = model(X[0].cuda(),X[1].cuda())
                #print(y_pre,y)
                loss = myloss(y_pre,y)
                loss_sum += loss.item()
                #acc_sum += (y_pre.argmax(dim=1)==y).sum().item()
                pre = y_pre>=torch.tensor([0.5]).cuda()
                corr = pre == y
                acc_sum += corr.sum().item()
                n+=1
                c+=y.shape[0]

            loss = loss_sum/n
            acc = acc_sum/c
            print(f'final test loss: {loss:.4f}, final test accuracy: {acc:.4f}')


def main(args):
    train_iter,test_iter = data_prepare(args)

    model = train(train_iter,args)

    eval(model,test_iter,args)

if __name__ == '__main__':
    args = get_parser()
    print(args)
    main(args)
    