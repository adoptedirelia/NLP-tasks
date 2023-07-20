import os   
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,6'  
  
import torch  
import torch.nn as nn   
import torch.optim as optim  
from torch.utils.data import DataLoader  
from torch.utils.data import Dataset  
from tqdm import tqdm  
import config  
import argparse  
from dataset import CONLLdata  
from model import MyBiLSTM_CRF  
from sklearn.metrics import precision_score  
from sklearn.metrics import accuracy_score  
from sklearn.metrics import recall_score  
from sklearn.metrics import f1_score  
from sklearn.metrics import classification_report  
import numpy as np  
import warnings  
warnings.filterwarnings("ignore")  
  
  
def get_parser():  
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification')  
    parser.add_argument('--config', type=str, default='./config.yaml', help='config file')  
    args = parser.parse_args()  
    assert args.config is not None  
    cfg = config.load_cfg_from_cfg_file(args.config)  
    return cfg  
  
def data_prepare(args):  
    train_data = CONLLdata()  
    pkg = train_data.getall()  
    test_data = CONLLdata(1,pkg)  
    val_data = CONLLdata(2,pkg)  
      
    train_iter = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)  
    test_iter = DataLoader(test_data,batch_size=args.batch_size)  
    val_iter = DataLoader(val_data,batch_size=args.batch_size)  
  
    return train_iter,test_iter,val_iter,pkg  
  
def train(train_iter,val_iter,voc_size,tags_size,idx_to_tag,args):  
    net = MyBiLSTM_CRF(voc_size=voc_size,tags_size=tags_size,emb_size=args.emb_size,hidden_size=args.hidden_size).cuda()  
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)  
    labels = idx_to_tag[1:]  
    print(labels)  
    final_score = np.zeros((8,3))  
    for epoch in range(args.epoch):  
        n,c,loss_sum,acc_sum = 0,0,0.0,0.0  
        score = np.zeros((8,3))  
        pres,re,f1s = 0.0,0.0,0.0  
        for X,y,mask in tqdm(train_iter):  
            X = X.cuda()  
            y = y.cuda()  
            mask = mask.cuda()  
            net.zero_grad()  
            loss = net(X,y,mask)  
            #print(loss.shape,X.shape,y.shape)  
            loss.backward()  
            optimizer.step()  
            n += 1  
            c += X.shape[0]  
            loss_sum += loss   
            with torch.no_grad():  
                path = net.seq(X,mask)  
            for y1,y2,m in zip(path,y,mask):  
                y1 = torch.tensor(y1,dtype=torch.int64).cpu()  
                y2 = y2[:m.sum()].cpu()  
                  
                pres += precision_score(y2,y1,average='macro')  
                re += recall_score(y2,y1,average='macro')   
                f1s += f1_score(y2,y1,average='macro')  
                my_pre = classification_report(y2,y1,output_dict=True)  
                #print(my_pre)  
  
                for i in my_pre:  
                    if not i.isdigit():  
                        continue  
                    ii = int(i)-1  
                    score[ii][0] += my_pre[i]['precision']  
                    score[ii][1] += my_pre[i]['recall']  
                    score[ii][2] += my_pre[i]['f1-score']  
        print(f"{epoch+1}\tprecision: {pres/c}\trecall: {re/c}\tf1-score: {f1s/c}")  
  
        print(f"{epoch+1}\tprecision\trecall\tf1-score")  
        for index,i in enumerate(score):  
            print(f"{labels[index]}\t{(i[0]/c):.4f}\t{(i[1]/c):.4f}\t{(i[2]/c):.4f}")  
        final_score += score/c  
  
    print(f"final:\tprecision\trecall\tf1-score")  
    for index,i in enumerate(final_score):  
        print(f"{labels[index]}\t{(i[0]/args.epoch):.4f}\t{(i[1]/args.epoch):.4f}\t{(i[2]/args.epoch):.4f}")  
  
    pre = final_score.sum(axis=0)[0]  
    recall = final_score.sum(axis=0)[1]  
    f1 = final_score.sum(axis=0)[2]  
    print(f"precision: {pre/args.epoch} recall: {recall/args.epoch} f1: {f1/args.epoch}")  
    return net   
  
def eval(model,test_iter,idx_to_tag,args):  
    labels = idx_to_tag[1:]  
  
    n,c,loss_sum,acc_sum = 0,0,0.0,0.0  
    pre_score,r_score,f_score = 0.0,0.0,0.0  
    score = np.zeros((8,3))  
    for X,y,mask in tqdm(test_iter):  
        X = X.cuda()  
        y = y.cuda()  
        mask = mask.cuda()  
        c += X.shape[0]  
        with torch.no_grad():  
            path = model.seq(X,mask)  
        for y1,y2,m in zip(path,y,mask):  
            y1 = torch.tensor(y1,dtype=torch.int64).cpu()  
            y2 = y2[:m.sum()].cpu()  
              
            my_pre = classification_report(y2,y1,output_dict=True)  
            #print(my_pre)  
  
            for i in my_pre:  
                if not i.isdigit():  
                    continue  
                ii = int(i)-1  
                score[ii][0] += my_pre[i]['precision']  
                score[ii][1] += my_pre[i]['recall']  
                score[ii][2] += my_pre[i]['f1-score']  
    print(f"\t\tprecision\trecall\tf1-score")  
    for index,i in enumerate(score):  
        print(f"{labels[index]}\t{(i[0]/c):.4f}\t{(i[1]/c):.4f}\t{(i[2]/c):.4f}")  
              
    print(f"precious {score.sum(axis=0)[0]/c} recall {score.sum(axis=0)[1]/c} f1 score {score.sum(axis=0)[2]/c}")  
      
  
def main(args):  
      
    train_iter,test_iter,val_iter,pkg = data_prepare(args)  
    print(pkg[0])  
    model = train(train_iter,val_iter,len(pkg[3]),len(pkg[1]),pkg[1],args)  
  
    eval(model,test_iter,pkg[1],args)  
  
if __name__ == '__main__':  
    args = get_parser()  
    print(args)  
    main(args)