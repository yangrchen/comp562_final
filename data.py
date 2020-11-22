import torch
import torch.nn as nn
from fastai.text import *
import numpy as np 
import torch.nn.functional as F
import token as tk
from components.vocab import *
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pickle
from tokenize import generate_tokens



def tokenize_code(code, mode='decoder'):
    
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    t=[0,4,5,6,54]
    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum not in t:
            tokens.append(tokval)

    return tokens


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

vocab = pickle.load(open("data/vocab.bin", 'rb'))

class Example:
    def __init__(self, input, output,target):
        self.input=input 
        self.output=output
        self.target=target

class MyDataset(Dataset):
    def __init__(self, df, max_len=25):
       
        self.examples=[]
        self.max_len=max_len
        self.df=df
        self.new_tensor = torch.LongTensor
        # self.zeros=self.new_tensor(self.max_len).zero_()

        self.init_dataset()
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]
    
    def init_dataset(self):
        a=self.df
        for i in range(len(a)):
            
            intent=a.iloc[i][0]
            snip=a.iloc[i][1]
            # print("intent is ",intent)
            """might need to change the tokenizer"""
            try:
                k=intent.split(' ')
            except:
                print(intent)
                continue
            
            try:
                t=tokenize_code(snip)
                # print(t)
            except:
                print(str(snip))
                continue

            if (len(k)>=self.max_len or len(t)>=self.max_len ):
                continue
            inputs=[]
            targets=[]
            for word in k:
                inputs.append(vocab.source[word])
            for token in t:
                targets.append(vocab.primitive[token])
            
            outputs=[0]+targets[:-1]
            inputs=torch.tensor(inputs)
            outputs=torch.tensor(outputs)
            targets=torch.tensor(targets)

            tokenized_inputs=self.new_tensor(self.max_len).zero_()
            tokenized_inputs[0:inputs.shape[0]]=inputs
           

            tokenized_targets=self.new_tensor(self.max_len).zero_()
            
            tokenized_targets[0:targets.shape[0]]=targets

            tokenized_outputs=self.new_tensor(self.max_len).zero_()
            
            tokenized_outputs[0:outputs.shape[0]]=outputs
           
            example=Example(tokenized_inputs,tokenized_outputs,tokenized_targets)
            self.examples.append(example)
    
    def batch_iter(self, bs):
        index=np.arange(len(self.examples))
        np.random.shuffle(index)
        
        num = int((len(self.examples) / float(bs))+0.5)
        for i in range(num):
            ids=index[bs*i:bs*(i+1)]
            ex=[self.examples[m] for m in ids]
            yield ex

