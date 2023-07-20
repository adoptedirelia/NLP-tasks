from torchcrf import CRF
import torch.nn as nn 
import torch 

class MyBiLSTM_CRF(nn.Module):
    def __init__(self,voc_size,tags_size,emb_size,hidden_size) -> None:
        super(MyBiLSTM_CRF,self).__init__()

        self.embedding = nn.Embedding(voc_size,emb_size)
        self.lstm = nn.LSTM(emb_size,hidden_size//2,num_layers=1,bidirectional=True,batch_first=True)
        
        self.hidden2tag = nn.Linear(hidden_size,tags_size)
        self.crf = CRF(tags_size,batch_first=True)

    def forward(self,X,tags,mask):
        emb = self.embedding(X)
        output,_ = self.lstm(emb)
        x = self.hidden2tag(output)
        
        loss = -self.crf(x,tags,mask)
        return loss 
    
    def seq(self,X,mask):
        emb = self.embedding(X)
        output,_ = self.lstm(emb)
        x = self.hidden2tag(output)

        return self.crf.decode(x,mask=mask)
    
if __name__ == '__main__':
    model = MyBiLSTM_CRF(21012,9,64,256)
    print(model)