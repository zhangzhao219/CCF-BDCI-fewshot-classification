import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertModel,BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from transformers import AutoModel, AutoConfig, AutoTokenizer

def getBert(bert_name):
    if bert_name == 'xlnet':
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    elif bert_name == 'ernie':
        print('load ernie-3.0-base-zh')
        model_config = AutoConfig.from_pretrained('nghuyong/ernie-3.0-base-zh')
        model_config.output_hidden_states = True
        bert = AutoModel.from_pretrained('nghuyong/ernie-3.0-base-zh', num_labels=36, output_attentions=False, output_hidden_states=False)
    elif bert_name == 'mengzi':
        print('load mengzi-bert-base')
        model_config = AutoConfig.from_pretrained('Langboat/mengzi-bert-base')
        model_config.output_hidden_states = True
        bert = AutoModel.from_pretrained('Langboat/mengzi-bert-base', num_labels=36, output_attentions=False, output_hidden_states=False)
    elif bert_name == 'bert_classification':
        print('load bert-base-uncased')
        # model_config = BertConfig.from_pretrained('bert-base-uncased')
        # model_config.output_hidden_states = True
        bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
    elif bert_name == 'ernie_classification':
        print('load ernie-3.0-base-zh for classification')
        model_config = AutoConfig.from_pretrained('nghuyong/ernie-3.0-base-zh')
        bert = BertForSequenceClassification.from_pretrained('nghuyong/ernie-3.0-base-zh', num_labels=36)
    else:
        print('No Bert Name!')
    return bert

def getTokenizer(bert_name):
    if bert_name == 'xlnet':
        print('load xlnet-base-cased tokenizer')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    elif bert_name == 'ernie':
        print('load ernie-3.0-base-zh tokenizer')
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
    elif bert_name == 'mengzi':
        print('load mengzi-bert-base tokenizer')
        tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-bert-base")
    elif bert_name == 'bert':
        print('load bert-base-uncased tokenizer')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    elif bert_name == 'ernie_classification':
        print('load ernie-3.0-base-zh tokenizer')
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
    else:
        print('No Tokenizer Name!')
    return tokenizer

class BertModel(nn.Module):
    def __init__(self, n_labels, bert='ernie', feature_layers=5, dropout=0.5):
        super(BertModel, self).__init__()
        self.bert = getBert(bert)
        # self.feature_layers = feature_layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.relu = nn.ReLU()

        # self.l0 = nn.Linear(self.feature_layers*self.bert.config.hidden_size, n_labels)

    def forward(self,input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_output = self.relu(linear_output)
        return final_output