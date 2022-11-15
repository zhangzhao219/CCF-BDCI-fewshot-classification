import numpy as np
from torch.utils.data import Dataset
class LoadData(Dataset):
    def __init__(self,data,tokenizer,max_len=512):
        super(LoadData,self).__init__()
        self.labels = data['label']
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        for text in data['sentence']:
            bert_inputs_dict = tokenizer(text, padding='max_length', max_length = max_len, truncation=True, return_tensors="pt")
            self.input_ids.append(bert_inputs_dict['input_ids'])
            self.token_type_ids.append(bert_inputs_dict['token_type_ids'])
            self.attention_mask.append(bert_inputs_dict['attention_mask'])
    def __getitem__(self,index):
        return (self.input_ids[index],self.token_type_ids[index] ,self.attention_mask[index]), np.array(self.labels[index])
    def __len__(self):
        return len(self.labels)