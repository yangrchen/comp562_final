from data import *
from model import *
from conala_eval import *
from bleu import *

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
  # print((pred==targ).float().mean(),"mean",(pred==targ).float().sum(),(pred==targ).float().shape)
  total=((pred==targ).float().shape[0])*(pred==targ).float().shape[1]
  # print("totals ",total, "corr ",corrects)
  return corrects,total


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
      # print("the type is ",type(outs[0]),outs[0])
      inp,out,targ=torch.stack(inps),torch.stack(outs),torch.stack(targ)
      pred=model(inp.to(device),out.to(device))
      # loss_func=nn.CrossEntropyLoss()
      loss = loss_func(pred.permute(0,2,1), targ.to(device))
      # print("train accuracy for this batch is: ",accuracy(pred, targ.to(device)))
      cor,tot=stats(pred, targ.to(device))
      corrects+=cor 
      totals+=tot
      train_loss+=loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    # print('train loss: ',train_loss)
    print("train_loss ",train_loss/j)
    print("train accuracy is: ",corrects/totals)
      # print("valid_loss ")
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
        # print(len(targ))

        inp,out,targ=torch.stack(inps),torch.stack(outs),torch.stack(targ)
        pred=model(inp.to(device),out.to(device))
        # print("valid accuracy for this batch is: ",accuracy(pred, targ.to(device)))
        cor,tot=stats(pred, targ.to(device))
        corrects+=cor 
        totals+=tot
        # print("the shape of pred",pred.shape)
        # print("the shape of targ",targ.shape)
        
        loss = loss_func(pred.permute(0,2,1), targ.to(device))
        # pred=torch.argmax(pred,dim=-1).squeeze()

        valid_loss+=loss
      print("valid_loss ",valid_loss/j)
      print("valid accuracy is: ",corrects/totals)
      model.train()
    special_tokens=['<unk>','<pad>']
    # if c_epoch>10:
    #   model.eval()
    #   i=0
    #   outputs=[]
    #   inputs=[]
    #   targets=[]
    #   for d in data.valid.batch_iter(bs=bs_test):
    #     # if i>0:
    #     #   break
    #     inps=[e.input for e in d]
    #     targs=[e.target for e in d]
    #     outs=[e.target for e in d]
    #     inp=torch.stack(inps)
    #     out=torch.stack(outs)
    #     pred=model(inp.to(device),out.to(device))
    #     # print("the shape of pred is ",pred)
    #     for m,j in enumerate(inps):
    #       o=decode(pred[m].argmax(dim=-1).squeeze().tolist())
    #       # print('the pred is ',pred[m])
    #       # print("the output is ",o)
    #       # o=decode(pred)
    #       t=decode(targs[m].tolist())
    #       outputs.append([x for x in o if x not in special_tokens])
    #       targets.append([x for x in t if x not in special_tokens])

    #       # outputs.append(' '.join([tokenizer_code.decode(pred)]))
    #       # targets.append(' '.join([tokenizer_code.decode(targ[m].tolist())]))
    #     i+=1
    #   # print("the lens are ",len(outputs),len(targets))
    #   print("the bleu score is ",evaluate(inputs,outputs,targets))
    #   print(len(outputs),len(targets))
    #   outputs=[{"code": o,"target":t} for o,t in zip(outputs,targets)]

    #   with open("infer/answer.txt", "w") as txt_file:
    #     json.dump(outputs,txt_file)
    #     # for line in outputs:
    #     #   txt_file.write(line + "\n")
    #   with open("infer/reference.txt", "w") as txt_file:
    #     json.dump(outputs,txt_file)
    #     # for line in targets:
    #     #   txt_file.write(line + "\n")

    #   model.train()
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
          # inputs.append(tokenizer_src.decode(j.tolist()))
          # print(tokenizer.decode(j.tolist()))
          # inp=torch.stack(inps)
          pred=model(j.to(device),gs=True)
          o=decode(pred)
          t=decode(targ[m].tolist())
          outputs.append([x for x in o if x not in special_tokens])
          targets.append([x for x in t if x not in special_tokens])

          # outputs.append(' '.join([tokenizer_code.decode(pred)]))
          # targets.append(' '.join([tokenizer_code.decode(targ[m].tolist())]))
        i+=1
      # print("the lens are ",len(outputs),len(targets))
      print("the bleu score is ",evaluate(inputs,outputs,targets))
      print(len(outputs),len(targets))
      outputs=[{"code": o,"target":t} for o,t in zip(outputs,targets)]

      with open("infer/answer.txt", "w") as txt_file:
        json.dump(outputs,txt_file)
        # for line in outputs:
        #   txt_file.write(line + "\n")
      with open("infer/reference.txt", "w") as txt_file:
        json.dump(outputs,txt_file)
        # for line in targets:
        #   txt_file.write(line + "\n")

      model.train()
    
    train_loss=valid_loss=0.

    if (c_epoch>=epoch):
      print("end of training")
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



args={'epoch':50}
train(args)












