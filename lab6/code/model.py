from transformers import BertModel, BertConfig

import torch.nn as nn 
import torch
configuration = BertConfig()
model = BertModel(configuration)
print(configuration)

class MyBert(nn.Module):
    def __init__(self,maxlen) -> None:
        super(MyBert,self).__init__()
        conf = BertConfig()
        conf.max_position_embeddings = maxlen
        print(conf)
        self.Bert = BertModel.from_pretrained('bert-base-cased')

        for name, param in self.Bert.named_parameters():
            if name.startswith('pooler'):
                continue
            else:
                param.requires_grad_(False)

        self.output = nn.Sequential(
            nn.Linear(768,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self,X,mask):
        feature = self.Bert(X,attention_mask=mask)
        media = feature['last_hidden_state'][:,0,:]
        media2 = feature['pooler_output']
        
        output = self.output(self.dropout(media2))
        
        return output