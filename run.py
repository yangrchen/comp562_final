from data import *
from model import *

class Data():
    def __init__(self, train, valid):
      self.train, self.valid=train, valid

# print(type(MyDataset))
def train(args):
  epoch=args['epoch']
  # tokenizer=AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
  # tokenizer.add_special_tokens({'bos_token':'<s>','eos_token':'</s>','unk_token':'unk'})
  """path of data"""
  a=pd.read_csv('data/a.csv')
  a=a.drop(columns=['Unnamed: 0'])
  train=a.sample(frac=0.8, random_state=42)
  valid=a.drop(train.index).reset_index(drop=True)

  train_dataset=MyDataset(tokenizer,train)
  valid_dataset=MyDataset(tokenizer,valid)

  bs_train=64;bs_valid=64

  # train_loader=DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
  # valid_loader=DataLoader(valid_dataset, batch_size=bs_valid, shuffle=True)

  

  data=Data(train_dataset,valid_dataset)



  model=Model().to(device)
  optimizer=torch.optim.Adam(params=model.parameters(), lr=3e-4)
  c_epoch=0
  while True:
    c_epoch+=1

    train_loss=valid_loss=0.
    j=0
    for d in data.train.batch_iter(bs=bs_train):
      j+=1
      inps=[e.input for e in d]
      outs=[e.target for e in d]
      # print("the type is ",type(outs[0]),outs[0])
      inp,out=torch.stack(inps),torch.stack(outs)
      pred=model(inp.to(device),out.to(device))
      loss_func=CrossEntropyFlat()
      loss = loss_func(pred, out.to(device))
      train_loss+=loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    print("train_loss ",train_loss/j)
      # print("valid_loss ")
    j=0
    torch.cuda.empty_cache()
    if c_epoch>1:
      for d in data.valid.batch_iter(bs=bs_valid):
        j+=1
        inps=[e.input for e in d]
        outs=[e.target for e in d]
        inp,out=torch.stack(inps),torch.stack(outs)
        pred=model(inp.to(device),out.to(device))
        loss_func=CrossEntropyFlat()
        loss = loss_func(pred, out.to(device))
        valid_loss+=loss
      print("valid_loss ",valid_loss/j)

    train_loss=valid_loss=0.

    if (c_epoch>=epoch):
      print("end of training")
      # model.save('last_model.bin')
      exit(0)

args={'epoch':5}
train(args)












