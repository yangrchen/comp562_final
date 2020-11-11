import torch
import torch.nn as nn
from fastai.text import *
import numpy as np 
import torch.nn.functional as F

try: 
    from transformers import *
except:
    print("install transformers first")
    # !pip install transformers
    from transformers import *

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Example:
    def __init__(self, input, target):
        self.input=input 
        self.target=target

class MyDataset(Dataset):
    def __init__(self, tokenizer, df, max_len=100):
        self.tokenizer=tokenizer 
        # self.inputs=[]
        # self.outputs=[]
        self.examples=[]
        self.max_len=max_len
        self.df=df

        self.init_dataset()
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]
    
    def init_dataset(self):
        a=self.df
        for i in range(len(a)):
            a.iloc[i][0]='<s> '+a.iloc[i][0]+' </s>'
            a.iloc[i][1]='<s> '+a.iloc[i][1]+' </s>'
            intent=a.iloc[i][0]
            snip=a.iloc[i][1]
            """might need to change the tokenizer"""
            tokenized_inputs=torch.tensor(self.tokenizer(intent, max_length=self.max_len,truncation='only_first',padding='max_length')['input_ids'])
            tokenized_outputs=torch.tensor(self.tokenizer(snip, max_length=self.max_len,truncation='only_first',padding='max_length')['input_ids'])

            example=Example(tokenized_inputs,tokenized_outputs)
            self.examples.append(example)
    
    def batch_iter(self, bs):
        index=np.arange(len(self.examples))
        np.random.shuffle(index)
        
        num = int((len(self.examples) / float(bs))+0.5)
        for i in range(num):
            ids=index[bs*i:bs*(i+1)]
            ex=[self.examples[m] for m in ids]
            yield ex

        


    


bs_train=64;bs_valid=64










