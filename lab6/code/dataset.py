import torch 
import torch.nn as nn 
import os 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import re
from transformers import AutoTokenizer



class MovieData(Dataset):
    def __init__(self,cls=0,maxlen = 2500) -> None:

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        path = ""
        if cls == 0:
            path = './aclImdb/train/'
        else:
            path = './aclImdb/test/'

        self.comments = []
        self.label = []

        items_pos = os.listdir(f'{path}pos')
        items_neg = os.listdir(f'{path}neg')

        print("读取积极评分...")

        for item in tqdm(items_pos):
            
            with open(f'{path}pos/{item}','r') as f:
                line = f.readline()
                line = line.strip().lower()
                encode_input = tokenizer(line,padding='max_length',truncation=True,max_length=maxlen,return_tensors='pt')
            

            self.comments.append((encode_input['input_ids'].squeeze(),encode_input['attention_mask'].squeeze()))
            self.label.append(torch.tensor([1]))

        print("读取消极评分...")

        for item in tqdm(items_neg):
            
            with open(f'{path}neg/{item}','r') as f:
                line = f.readline()
                line = line.strip().lower()
                encode_input = tokenizer(line,padding='max_length',truncation=True,max_length=maxlen,return_tensors='pt')

            self.comments.append((encode_input['input_ids'].squeeze(),encode_input['attention_mask'].squeeze()))
            self.label.append(torch.tensor([0]))
    

    def __getitem__(self, index):
        return self.comments[index],self.label[index]
    
    def __len__(self):
        return len(self.comments)





