import numpy as np
from torch.utils.data import Dataset
class LoadData(Dataset):
    def __init__(self,data,tokenizer,max_len=512):
        super(LoadData,self).__init__()
        self.labels = data['label']
        self.texts = [tokenizer(
                            text,
                            padding='max_length',
                            max_length = max_len, 
                            truncation=True,
                            return_tensors="pt"
                            ) for text in data['sentence']]
    def __getitem__(self,index):
        return self.texts[index],np.array(self.labels[index])
    def __len__(self):
        return len(self.labels)