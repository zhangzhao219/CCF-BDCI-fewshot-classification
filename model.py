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

class RDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """
    def __init__(self):
        super(RDrop, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')

    def forward(self, logits1, logits2, target, kl_weight=1.):
        """
        Args:
            logits1: One output of the classification model.
            logits2: Another output of the classification model.
            target: The target labels.
            kl_weight: The weight for `kl_loss`.

        Returns:
            loss: Losses with the size of the batch size.
        """
        ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + kl_weight * kl_loss
        return loss.mean()

# FGM
class FGM:
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}
    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}

# PGD
class PGD:
    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
    def attack(self, emb_name='embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)
    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
# EMA
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self, warmup_if=False):
        decay = 0.0 if warmup_if else self.decay
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
