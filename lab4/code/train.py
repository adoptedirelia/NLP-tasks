import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'

import torch
import torch.nn as nn 
import Moviedataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import config
import argparse
from model import CNN

def get_parser():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification')
    parser.add_argument('--config', type=str, default='./config.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def prepare_data(args):

    trainset = Moviedataset.MovieData(0)
    testset = Moviedataset.MovieData(1)
    
    train_iter = DataLoader(trainset,args.batch_size,shuffle=True)
    test_iter = DataLoader(testset,args.batch_size)
    print(f"词汇表大小: {trainset.voc_len()}")
    return train_iter,test_iter

def train(train_iter,args):
    net = CNN(89529,args.emb_sz).cuda()
    print(net)
    net = nn.DataParallel(net)
    myloss = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(),args.lr)
    net.train()
    for epoch in range(args.epoch):
        loss_sum,n,acc_sum,c = 0.0,0,0.0,0
        with tqdm(train_iter) as t:
            for X,y in t:
                t.set_description(f"epoch {epoch+1}")
                X[0] = X[0].cuda()
                y = y.cuda().float()
                optimizer.zero_grad()

                # forward + backward + optimize
                y_pre = net(X[0])
                #print(y_pre)
                loss = myloss(y_pre,y)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                pre = y_pre>=torch.tensor([0.5]).cuda()
                corr = pre == y
                acc_sum += corr.sum().item()
                n+=1
                c+=y.shape[0]
            loss = loss_sum/n
            acc = acc_sum/c
            print(f'epoch {epoch+1}: loss: {loss:.4f} accuracy: {acc:.4f}')
    
    return net

def eval(model,test_iter,args):

    myloss = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        loss_sum,n,acc_sum,c = 0.0,0,0.0,0
        with tqdm(test_iter) as t:
            for X,y in t:
                X[0] = X[0].cuda()
                y = y.cuda().float()
                y_pre = model(X[0])
                loss = myloss(y_pre,y)
                loss_sum += loss.item()
                pre = y_pre>=torch.tensor([0.5]).cuda()
                corr = pre == y
                acc_sum += corr.sum().item()
                n+=1
                c+=y.shape[0]

            loss = loss_sum/n
            acc = acc_sum/c
            print(f'final test loss: {loss:.4f}, final test accuracy: {acc:.4f}')


def main(args):
    train_iter,test_iter = prepare_data(args)

    model = train(train_iter,args)

    eval(model,test_iter,args)
    
    

if __name__ == '__main__':
    args = get_parser()
    print(args)
    main(args)
