from data import *
from model import *
from conala_eval import *
from bleu import *
import matplotlib.pyplot as plt

class Data():
    def __init__(self, train, valid,test=None):
      self.train, self.valid=train, valid
      self.test=test

def decode(tokens,source=False):
  out=[]
  if not source:
    for t in tokens:
      out.append(vocab.primitive.id2word[t])
  else:
    for t in tokens:
      out.append(vocab.source.id2word[t])
  return out

def stats(pred, targ):
  pred=pred.argmax(dim=-1).squeeze()
  corrects=(pred==targ).float().sum()
  total=((pred==targ).float().shape[0])*(pred==targ).float().shape[1]
  return corrects,total


def train(args):
  epoch=args['epoch']
  """path of data"""
  a=pd.read_csv('data/a.csv')
  a=a.drop(columns=['Unnamed: 0'])
  train=a.sample(frac=0.8, random_state=42)
  valid=a.drop(train.index).reset_index(drop=True)
  print ("the length of valid set is ",len(valid))
  test=pd.read_csv('test.csv')
  test=test.drop(columns=['Unnamed: 0'])

  train_dataset=MyDataset(train)
  valid_dataset=MyDataset(valid)
  test_dataset=MyDataset(test)
  print(len(test_dataset), 'this is the len of test')

  bs_train=64;bs_valid=64;bs_test=64

  
  data=Data(train_dataset,valid_dataset,test_dataset)



  model=Model().to(device)
  optimizer=torch.optim.Adam(params=model.parameters(), lr=3e-4)
  c_epoch=0
  loss_func=nn.CrossEntropyLoss()
  train_loss_values=[]
  valid_loss_values=[]
  while True:
    c_epoch+=1

    train_loss=valid_loss=0.
    j=0
    print("the len of train is",len(data.valid))
    totals,corrects=0,0
    for d in data.train.batch_iter(bs=bs_train):
      
      j+=1
      inps=[e.input for e in d]
      outs=[e.output for e in d]
      targ=[e.target for e in d]
      inp,out,targ=torch.stack(inps),torch.stack(outs),torch.stack(targ)
      pred=model(inp.to(device),out.to(device))
      loss = loss_func(pred.permute(0,2,1), targ.to(device))
      cor,tot=stats(pred, targ.to(device))
      corrects+=cor 
      totals+=tot
      train_loss+=loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    print("train_loss ",train_loss/j)
    train_loss_values.append(train_loss/j)
    print("train accuracy is: ",corrects/totals)
    j=0
    torch.cuda.empty_cache()
    totals,corrects=0,0
    if c_epoch>0:
      model.eval()
      for d in data.valid.batch_iter(bs=bs_valid):
        j+=1
        inps=[e.input for e in d]
        outs=[e.output for e in d]
        targ=[e.target for e in d]

        inp,out,targ=torch.stack(inps),torch.stack(outs),torch.stack(targ)
        pred=model(inp.to(device),out.to(device))
        cor,tot=stats(pred, targ.to(device))
        corrects+=cor 
        totals+=tot
        
        loss = loss_func(pred.permute(0,2,1), targ.to(device))

        valid_loss+=loss
      print("valid_loss ",valid_loss/j)
      valid_loss_values.append(valid_loss/j)
      print("valid accuracy is: ",corrects/totals)
      model.train()
    special_tokens=['<unk>','<pad>']
    
    if c_epoch>18:
      model.eval()
      i=0
      outputs=[]
      inputs=[]
      targets=[]
      for d in data.test.batch_iter(bs=bs_test):
        # if i>2:
        #   break
        inp=[e.input for e in d]
        targ=[e.target for e in d]
        for m,j in enumerate(inp):
          
          pred=model(j.to(device),gs=True)
          o=decode(pred)
          t=decode(targ[m].tolist())
          outputs.append([x for x in o if x not in special_tokens])
          targets.append([x for x in t if x not in special_tokens])

          
        i+=1
      print("the bleu score is ",evaluate(inputs,outputs,targets))
      print(len(outputs),len(targets))
      outputs=[{"code": o,"target":t} for o,t in zip(outputs,targets)]

      with open("infer/answer.txt", "w") as txt_file:
        json.dump(outputs,txt_file)
        
      with open("infer/reference.txt", "w") as txt_file:
        json.dump(outputs,txt_file)
        # for line in targets:
        #   txt_file.write(line + "\n")

      model.train()
    
    train_loss=valid_loss=0.

    if (c_epoch>=epoch):
      print("end of training")
      print("the loss values are: ")
      plt.plot(train_loss_values)
      plt.plot(valid_loss_values)
      print(train_loss_values," train loss")
      print(valid_loss_values," valid loss")

      # model.save('last_model.bin')
      exit(0)

def evaluate(inputs, outputs, targets):
  print(outputs)
  print("targets ",targets)
  for i in range(10):
    print("output: ",outputs[i])
    print("target: ",targets[i])
  bleu_tup = compute_bleu([[tokenize_for_bleu_eval(''.join(x))] for x in outputs], [tokenize_for_bleu_eval(''.join(x)) for x in targets], smooth=False)
  # bleu_tup = compute_bleu([[x] for x in outputs], targets, smooth=False)

  bleu = bleu_tup[0]
  print(bleu_tup)
  return bleu



args={'epoch':5}
train(args)












