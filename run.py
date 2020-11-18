from data import *
from model import *
from conala_eval import *
from bleu import *

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
  print ("the length of valid set is ",len(valid))

  train_dataset=MyDataset(tokenizer_src,tokenizer_code,train)
  valid_dataset=MyDataset(tokenizer_src,tokenizer_code,valid)

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
      outs=[e.output for e in d]
      targ=[e.target for e in d]
      # print("the type is ",type(outs[0]),outs[0])
      inp,out,targ=torch.stack(inps),torch.stack(outs),torch.stack(targ)
      pred=model(inp.to(device),out.to(device))
      loss_func=CrossEntropyFlat()
      loss = loss_func(pred, targ.to(device))
      train_loss+=loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    print("train_loss ",train_loss/j)
      # print("valid_loss ")
    j=0
    torch.cuda.empty_cache()
    if c_epoch>0:
      model.eval()
      for d in data.valid.batch_iter(bs=bs_valid):
        j+=1
        inps=[e.input for e in d]
        outs=[e.target for e in d]
        targ=[e.target for e in d]
        # print(len(targ))

        inp,out,targ=torch.stack(inps),torch.stack(outs),torch.stack(targ)
        pred=model(inp.to(device),out.to(device))
        loss_func=CrossEntropyFlat()
        loss = loss_func(pred, targ.to(device))
        # pred=torch.argmax(pred,dim=-1).squeeze()

        valid_loss+=loss
      print("valid_loss ",valid_loss/j)
      model.train()
    
    if c_epoch==9:
      model.eval()
      i=0
      outputs=[]
      inputs=[]
      targets=[]
      for d in data.valid.batch_iter(bs=bs_valid):
        if i>0:
          break
        inp=[e.input for e in d]
        targ=[e.target for e in d]
        for m,j in enumerate(inp):
          inputs.append(tokenizer_src.decode(j.tolist()))
          # print(tokenizer.decode(j.tolist()))
          # inp=torch.stack(inps)
          pred=model(j.to(device),gs=True)
          outputs.append(' '.join([tokenizer_code.decode(pred)]))
          targets.append(' '.join([tokenizer_code.decode(targ[m].tolist())]))
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
  bleu_tup = compute_bleu([[x] for x in outputs], targets, smooth=False)
  bleu = bleu_tup[0]
  print(bleu_tup)
  return bleu



args={'epoch':10}
train(args)












