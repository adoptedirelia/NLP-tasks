import torch.nn.functional as F
import torch.nn as nn 
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,voc,emb_sz,drop_rate=0.5) -> None:
        super(CNN,self).__init__()
        
        self.embedding = nn.Embedding(voc,emb_sz)


        self.conv1 = nn.Sequential(
            nn.Conv2d(1,1,(3,emb_sz)),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1,1,(4,emb_sz)),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1,1,(5,emb_sz)),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1,1,(6,emb_sz)),
            nn.ReLU(),
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool2d((2500-3,1)),
            nn.ReLU()
        )

        self.pool2 = nn.Sequential(
            nn.MaxPool2d((2500-4,1)),
            nn.ReLU()
        )

        self.pool3 = nn.Sequential(
            nn.MaxPool2d((2500-5,1)),
            nn.ReLU()
        )

        self.pool4 = nn.Sequential(
            nn.MaxPool2d((2500-6,1)),
            nn.ReLU()
        )
        
        self.linear = nn.Linear(4,1)
        self.dropout = nn.Dropout(drop_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self,X):
        X = self.embedding(X)
        X = X.unsqueeze(1)

        x1 = self.dropout(self.conv1(X))
        x2 = self.dropout(self.conv2(X))
        x3 = self.dropout(self.conv3(X))
        x4 = self.dropout(self.conv4(X))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        x4 = self.pool4(x4)


        x = torch.cat((x1,x2,x3,x4),-1)
        x = x.view(X.shape[0],-1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x