import torch
import torch.nn as nn

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

class PretrainedModel(nn.Module):
    def __init__(self, bert, n_labels, feature_layers, dropout):
        super(PretrainedModel, self).__init__()
        self.bert = getBert(bert)
        self.feature_layers = feature_layers
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(self.bert.config.hidden_size * feature_layers, n_labels)
        self.omega = torch.nn.Parameter(torch.ones(1,n_labels))
        self.beta = torch.nn.Parameter(torch.zeros(1,n_labels))
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
        return self.linear(self.dropout(output)), self.omega, self.beta