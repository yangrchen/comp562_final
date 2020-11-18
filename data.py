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
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer_src = Tokenizer(BPE())
trainer_src = BpeTrainer(vocab_size=2000,special_tokens=["unk"])
tokenizer_src.train(trainer_src, ["intent.txt"])

tokenizer_code = Tokenizer(BPE())
trainer_code = BpeTrainer(vocab_size=2000,special_tokens=["unk"])
tokenizer_code.train(trainer_code, ["snip.txt"])

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# class tokenizer
class Example:
    def __init__(self, input, output,target):
        self.input=input 
        self.output=output
        self.target=target

class MyDataset(Dataset):
    def __init__(self, tokenizer_src,tokenizer_code, df, max_len=30):
        self.tokenizer_src=tokenizer_src
        self.tokenizer_code=tokenizer_code
        # self.inputs=[]
        # self.outputs=[]
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
            a.iloc[i][0]=a.iloc[i][0]
            a.iloc[i][1]=a.iloc[i][1]
            intent=a.iloc[i][0]
            snip=a.iloc[i][1]
            """might need to change the tokenizer"""
            inputs=torch.tensor(self.tokenizer_src.encode(intent).ids)
            targets=torch.tensor(self.tokenizer_code.encode(snip).ids)
            outputs=torch.tensor([0]+targets.tolist()[:-1])
            # outputs=torch.tensor([0]+self.tokenizer.encode(snip).ids[:-1])

            if (inputs.shape[0]>=self.max_len or outputs.shape[0]>=self.max_len or targets.shape[0]>=self.max_len):
                # print(tokenized_inputs.shape[0],tokenized_outputs.shape[0])
                continue

            # self.tokenizer.enable_padding(length=self.max_len)
            tokenized_inputs=self.new_tensor(self.max_len).zero_()
            # print(tokenized_inputs,"zeros")
            
            tokenized_inputs[0:inputs.shape[0]]=inputs
            
            # print("shape ",inputs.shape[0])
            # print(inputs, "inputs")
            # print(tokenized_inputs,"tok inputs")

            tokenized_targets=self.new_tensor(self.max_len).zero_()
            
            tokenized_targets[0:targets.shape[0]]=targets

            tokenized_outputs=self.new_tensor(self.max_len).zero_()
            
            tokenized_outputs[0:outputs.shape[0]]=outputs
            # print(outputs,"output")
            # print(targets,"targets")

            
            

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

        


    


bs_train=64;bs_valid=64










