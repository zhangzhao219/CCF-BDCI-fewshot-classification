from typing import List, Optional
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, DataProcessor
from openprompt.data_utils.utils import InputExample
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer
from openprompt.prompts.manual_verbalizer import ManualVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
import torch 
from torch.optim import AdamW
from prompt_dict import get_label_words_list
import pandas as pd

class CustomProcessor(DataProcessor):
    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:

        path = os.path.join(data_dir, "{}.json".format(split))
        examples = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                data_point = json.loads(line)
                example = InputExample(
                    meta={"title": data_point["title"], "assignee": data_point["assignee"], "abstract": data_point["abstract"]},
                    label=int(data_point["label_id"])
                )
                examples.append(example)

        return examples

        
dataset = {}
dataset['train'] = CustomProcessor().get_examples('./data/fewshot', 'expand_train_cur_best')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline, BertTokenizer, MT5ForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("uer/t5-v1_1-base-chinese-cluecorpussmall")
model = MT5ForConditionalGeneration.from_pretrained("uer/t5-v1_1-base-chinese-cluecorpussmall")



from openprompt.plms.seq2seq import T5TokenizerWrapper

# wrapped_tokenizer = T5TokenizerWrapper(max_seq_length=512, tokenizer=tokenizer, decoder_max_length=5, decode_from_pad=False)

class T5BertTokenizerWrapper(T5TokenizerWrapper):
    def mask_token(self, id):
        return f"extra{id}"

    def mask_token_ids(self, id):
        return self.tokenizer.convert_tokens_to_ids(f"extra{id}")
tokenizer.eos_token = "extra1"

tokenizer_wrapper = T5BertTokenizerWrapper(max_seq_length=512, tokenizer=tokenizer, decoder_max_length=5, decode_from_pad=False)



from openprompt.prompts import ManualTemplate
choice = ''
custom_template = ManualTemplate(tokenizer=tokenizer, text="""{"meta": "abstract", "shortenable": "True"}。上述文本的话题是{"mask"}""")
wrapped_example = custom_template.wrap_one_example(dataset['train'][0])
print(wrapped_example)

custom_verbalizer = ManualVerbalizer(tokenizer, num_classes=36, label_words=get_label_words_list())



train_dataloader = PromptDataLoader(dataset=dataset["train"], template=custom_template, verbalizer=custom_verbalizer, # be sure to add verbalizer
    tokenizer_wrapper=tokenizer_wrapper,
    batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=True,
    truncate_method="tail")





use_cuda = True
prompt_model = PromptForClassification(plm=model,template=custom_template, verbalizer=custom_verbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']


optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
generation_arguments = {
    "max_length": 2,
}



def evaluate(prompt_model, dataloader):
    prompt_model.eval()

    allpreds = []
    alllabels = []
    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print("validation:",acc)

#training with full data
for epoch in range(5):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(tot_loss/(step+1))
            # evaluate(prompt_model, val_dataloader)


#test
class TestProcessor(DataProcessor):
    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:

        path = os.path.join(data_dir, "{}.json".format(split))
        examples = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                data_point = json.loads(line)
                example = InputExample(
                    meta={"title": data_point["title"], "assignee": data_point["assignee"], "abstract": data_point["abstract"]},
                )
                examples.append(example)

        return examples

dataset['test'] = TestProcessor().get_test_examples('./data/fewshot')
test_dataloader =  PromptDataLoader(dataset=dataset["test"], template=custom_template, verbalizer=custom_verbalizer, # be sure to add verbalizer
    tokenizer_wrapper=tokenizer_wrapper,
    batch_size=8, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

prompt_model.eval()
allpreds = []
for step, inputs in enumerate(test_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

result = {"pred": allpreds}
pred = pd.DataFrame(data=result)
pred.to_csv('result.csv', index=False)

 