import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import AutoModel, AutoConfig, AutoTokenizer

def getBert(bert_name):
    print('load '+ bert_name)
    model_config = AutoConfig.from_pretrained(bert_name)
    model_config.output_hidden_states = True
    bert = AutoModel.from_pretrained(bert_name,config=model_config)
    return bert

def getTokenizer(bert_name):
    print('load '+ bert_name + ' tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    return tokenizer

class BertModel(nn.Module):
    def __init__(self, bert, n_labels, feature_layers, dropout):
        super(BertModel, self).__init__()
        self.bert = getBert(bert)
        self.feature_layers = feature_layers
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(self.bert.config.hidden_size * feature_layers, n_labels)
 
        # nn.init.normal_(self.linear.weight, std=0.02)
        # nn.init.normal_(self.linear.bias, 0)
        self.linear = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, n_labels)
        )

    def forward(self,input_ids,token_type_ids,attention_mask):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # print(outputs['pooler_output'].shape)
        # print(outputs['last_hidden_state'].shape)
        # print(outputs['hidden_states'].shape)
        # output = torch.cat([outputs['hidden_states'][-i][:, 0] for i in range(1, self.feature_layers+1)], dim=-1)
        output = torch.mean(outputs['last_hidden_state'], 1)
        return self.linear(self.dropout(output))
    
    def forward_mix_encoder(self, x1, att1, x2, att2, token_type_ids, lam):
        outputs1 = self.bert(input_ids=x1, token_type_ids=token_type_ids, attention_mask=att1)
        outputs2 = self.bert(input_ids=x2, token_type_ids=token_type_ids, attention_mask=att2)
        output1 = torch.mean(outputs1['last_hidden_state'], 1)
        output2 = torch.mean(outputs2['last_hidden_state'], 1)
        pooled_output = lam * output1 + (1.0 - lam) * output2
        y = self.linear(self.dropout(pooled_output))
        return y
