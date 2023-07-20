import torch 
import torch.nn as nn 
import os 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import re

class MovieData(Dataset):
    def __init__(self,cls=0,maxlen = 2500) -> None:
        self.voc = Vocab()

        path = ""
        if cls == 0:
            path = './aclImdb/train/'
        else:
            path = './aclImdb/test/'

        self.comments = []
        self.scores = []
        self.token_to_idx,self.idx_to_token = self.voc.getvoc()
        self.maxlen = maxlen
        items_pos = os.listdir(f'{path}pos')
        items_neg = os.listdir(f'{path}neg')

        print("读取积极评分...")

        for item in tqdm(items_pos):
            score = int(item.split('_')[1].split('.')[0])
            with open(f'{path}pos/{item}','r') as f:

                comment = f.read()
                comment = re.sub(r'[,.!\s\"?<>/:]+',' ',comment).strip()
                comment = re.sub(r'[\']+','',comment).strip()


            comment = comment.lower()
            words = comment.split(" ")
            lst = []
            for word in words:

                if self.token_to_idx.get(word) == None:
                    lst.append(1)
                else:
                    lst.append(self.token_to_idx[word])

            lst += [0] * (self.maxlen-len(lst))

            self.comments.append((torch.tensor(lst),comment))
            self.scores.append(torch.tensor([1]))

        print("读取消极评分...")

        
        for item in tqdm(items_neg):
            score = int(item.split('_')[1].split('.')[0])

            with open(f'{path}neg/{item}','r') as f:

                comment = f.read()
                comment = re.sub(r'[,.!\s\"?<>/:]+',' ',comment).strip()
                comment = re.sub(r'[\']+','',comment).strip()

            comment = comment.lower()
            words = comment.split(" ")
            lst = []
            for word in words:
                if self.token_to_idx.get(word) == None:
                    lst.append(1)
                else:
                    lst.append(self.token_to_idx[word])
            lst += [0] * (self.maxlen-len(lst))


            self.comments.append((torch.tensor(lst),comment))
            self.scores.append(torch.tensor([0]))

    def __getitem__(self, index):
        return self.comments[index],self.scores[index]
    
    def __len__(self):
        return len(self.comments)
    
    def voc_len(self):
        return len(self.voc)


class Vocab:
    def __init__(self) -> None:
        path = './aclImdb/imdb.vocab'
        file = open(path,'r')
        lines = file.readlines()

        self.token_to_idx = {'<pad>':0,'<unk>':1}
        self.idx_to_token = ['<pad','<unk>']

        index = 2
        for line in lines:
            line = line.strip()
            self.token_to_idx[line] = index 
            index += 1
            self.idx_to_token.append(line)

    def getvoc(self):
        return self.token_to_idx,self.idx_to_token
    
    def __len__(self):
        return len(self.idx_to_token)
    


