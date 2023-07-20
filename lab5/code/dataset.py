import torch 
import torch.nn as nn 
import os 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import re

class CONLLdata(Dataset):
    def __init__(self,cls=0,pkg=None):
        file = None
        if cls == 0:
            file = open('./data/eng.train','r')
        elif cls ==1:
            file = open('./data/eng.testa','r')
        else:
            file = open('./data/eng.testb','r')
        if pkg != None:
            self.label_to_idx,self.idx_to_label,self.token_to_idx,self.idx_to_token = pkg[0],pkg[1],pkg[2],pkg[3]
        else:
            voc = Vocab()
            self.label_to_idx,self.idx_to_label = voc.getlabel()
            self.token_to_idx,self.idx_to_token = voc.getvoc()
        self.sent = []
        self.sen_label = []
        self.mask = []
        maxlen = 115
        lst1 = []
        lst2 = []
        lines = file.readlines()
        for line in tqdm(lines):
            #print(line)

            if line =="\n": # 一句话结束了
                mask = [1]*len(lst1) + [0]*(maxlen-len(lst1))
                lst1 += [0] *(maxlen-len(lst1))
                lst2 += [0] *(maxlen-len(lst2))
                
                self.mask.append(torch.tensor(mask,dtype=torch.bool))
                self.sent.append(torch.tensor(lst1))
                self.sen_label.append(torch.tensor(lst2,dtype=torch.int64))
                
                lst1 = []
                lst2 = []
                continue
            line = line.strip().lower()
            words = line.split(" ")
            if words[0] not in self.idx_to_token:
                lst1.append(self.token_to_idx['<unk>'])
            else:
                lst1.append(self.token_to_idx[words[0]])
            #print(lst2,words)
            lst2.append(self.label_to_idx[words[-1]])


    def __getitem__(self, index):
        return self.sent[index],self.sen_label[index],self.mask[index]
    
    def __len__(self):
        return len(self.sent)
    
    def getall(self):
        return (self.label_to_idx,self.idx_to_label,self.token_to_idx,self.idx_to_token)


class Vocab:
    def __init__(self) -> None:
        path = './data/eng.train'
        file = open(path,'r')
        lines = file.readlines()

        self.token_to_idx = {'<pad>':0,'<unk>':1}
        self.idx_to_token = ['<pad>','<unk>']
        self.label_to_idx = {'<pad>':0}
        self.idx_to_label = ['<pad>']

        index_word = 2
        index_label = 1
        for line in tqdm(lines):
            if line == "\n":
                continue
            line = line.strip().lower()
            
            words = line.split(" ")

            if words[0] not in self.idx_to_token:
                self.idx_to_token.append(words[0])
                self.token_to_idx[words[0]] = index_word
                index_word += 1
                
            if words[-1] not in self.idx_to_label:
                self.idx_to_label.append(words[-1])
                self.label_to_idx[words[-1]] = index_label
                index_label += 1

    def getvoc(self):
        return self.token_to_idx,self.idx_to_token
    
    def getlabel(self):
        return self.label_to_idx,self.idx_to_label
    
    def __len__(self):
        return len(self.idx_to_token)

