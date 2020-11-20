## **Pre**

# !cp /content/drive/My\ Drive/code\ generation/conala-corpus/a.csv /content

from fastai.text import *

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# !pip install transformers

from transformers import *
from conala_eval import *
from bleu import *

## **Data**



tokenizer=AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

tokenizer.add_special_tokens({'bos_token':'<s>','eos_token':'</s>','unk_token':'unk'})




a=pd.read_csv('data/a.csv')

a.head()

a=a.drop(columns=['Unnamed: 0'])

train=a.sample(frac=0.8, random_state=42)

valid=a.drop(train.index).reset_index(drop=True)

valid=valid.sample(frac=0.1, random_state=42).reset_index(drop=True)

len(train)

train=train.reset_index(drop=True)

class MyDateset(Dataset):
  def __init__(self, tokenizer, df,max_len=20):
    self.tokenizer=tokenizer
    self.inputs=[]
    self.targets=[]
    self.outputs=[]
    self.max_len=max_len
    self.df=df

    self._build()

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    # source_ids=self.inputs[index]['input_ids'].squeeze()
    # target_ids=self.targets[index]['input_ids'].squeeze()
    # src_mask = self.inputs[index]["attention_mask"].squeeze()
    # target_mask = self.targets[index]["attention_mask"].squeeze()

    # return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
    source_ids=torch.tensor(self.inputs[index]['input_ids'])
    output_ids=torch.tensor(self.outputs[index]['input_ids'])
    target_ids=torch.tensor(self.targets[index]['input_ids'])
    return {'source_ids': source_ids,'output_ids':output_ids,'target_ids': target_ids}
  

  def _build(self):
    a=self.df
    for i in range(len(a)-1):
    #   a.iloc[i][0]=a.iloc[i][0]
    #   a.iloc[i][1]=a.iloc[i][1]
      intent=a.iloc[i][0]
      snip=a.iloc[i][1]
      # print(intent)
      # print(snip)
      # tokenized_inputs=self.tokenizer.encode_plus(intent, max_length=self.max_len,pad_to_max_length=True, return_tensors='pt')
      # tokenized_outputs=self.tokenizer.encode_plus(snip, max_length=self.max_len,pad_to_max_length=True, return_tensors='pt')
      tokenized_inputs=self.tokenizer(intent, max_length=self.max_len,truncation='only_first',padding='max_length')
      
      tokenized_targets=self.tokenizer(snip, max_length=self.max_len,truncation='only_first',padding='max_length')
      tokenized_outputs={'input_ids':[]}
      tokenized_outputs['input_ids']=[0]+tokenized_targets['input_ids'][:-1]
      # print(tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids']))
      # print(tokenizer.convert_ids_to_tokens(tokenized_outputs['input_ids']))


      self.inputs.append(tokenized_inputs)
      self.outputs.append(tokenized_outputs)
      self.targets.append(tokenized_targets)

    





# train_dataset[1]

train_dataset=MyDateset(tokenizer,train)

valid_dataset=MyDateset(tokenizer,valid)

bs_train=64;bs_valid=64

train_loader=DataLoader(train_dataset, batch_size=bs_train,shuffle=True,num_workers=4)

valid_loader=DataLoader(valid_dataset, batch_size=bs_valid,shuffle=True,num_workers=4)

# train_loader=load.train_dataloader()

# valid_loader=load.valid_dataloader()



class Data():
  def __init__(self, train_dl, valid_dl):
    self.train_dl, self.valid_dl=train_dl, valid_dl 

data=Data(train_loader,valid_loader)

len(data.train_dl)

## **BLEU**


## **Model**

model_hugging=AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")

inp_emb=model_hugging.get_input_embeddings()

v_size=tokenizer.vocab_size
print("v_size is ",v_size)

class PositionalEncoding(nn.Module):
    "Encode the position with a sinusoid."
    def __init__(self, d):
        super().__init__()
        self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.)/d)))
    
    def forward(self, pos):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc


class TransformerEmbedding(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self,inp_emb=inp_emb, emb_sz=768, inp_p=0.):
        super().__init__()
        self.emb_sz = emb_sz
        # self.embed = embedding(vocab_sz, emb_sz)
        self.embed=inp_emb
        self.pos_enc = PositionalEncoding(self.emb_sz)
        self.drop = nn.Dropout(inp_p)
    
    def forward(self, inp): 
        pos = torch.arange(0, inp.size(1), device=inp.device).float()
        return self.drop(self.embed(inp) * math.sqrt(self.emb_sz) + self.pos_enc(pos))



def feed_forward(d_model, d_ff, ff_p=0., double_drop=True):
    layers = [nn.Linear(d_model, d_ff), nn.ReLU()]
    if double_drop: layers.append(nn.Dropout(ff_p))
    return SequentialEx(*layers, nn.Linear(d_ff, d_model), nn.Dropout(ff_p), MergeLayer(), nn.LayerNorm(d_model))

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_head=None, p=0., bias=True, scale=True):
        super().__init__()
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        self.q_wgt,self.k_wgt,self.v_wgt = [nn.Linear(
            d_model, n_heads * d_head, bias=bias) for o in range(3)]
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att,self.drop_res = nn.Dropout(p),nn.Dropout(p)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, q, kv, mask=None):
        return self.ln(q + self.drop_res(self.out(self._apply_attention(q, kv, mask=mask))))
    
    def create_attn_mat(self, x, layer, bs):
        return layer(x).view(bs, x.size(1), self.n_heads, self.d_head
                            ).permute(0, 2, 1, 3)
    
    def _apply_attention(self, q, kv, mask=None):
        # if mask is not None:
          
        # print ('original shapes:','q:',q.shape,'kv: ',kv.shape)
        bs,seq_len = q.size(0),q.size(1)
        wq,wk,wv = map(lambda o: self.create_attn_mat(*o,bs),
                       zip((q,kv,kv),(self.q_wgt,self.k_wgt,self.v_wgt)))
        # print ('heads shapes:','q:',wq.shape,'kv: ',wk.shape,wv.shape)
        attn_score = wq @ wk.transpose(2,3)
        if self.scale: attn_score /= math.sqrt(self.d_head)
        # print ('orig attn_score: ',attn_score.shape)
        if mask is not None: 
            # print ('mask',mask.shape)
            # print ('orig: ',attn_score.shape )
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
            # print ('masked: ',attn_score.shape)
        # print ('masked attn_score: ',attn_score.shape)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        # print ('attn_prob',attn_prob.shape)
        attn_vec = attn_prob @ wv
        # print ('orig attn_vec:',attn_vec.shape)
        # print ('final',attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1).shape)
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)


def get_output_mask(inp, pad_idx=1):
    return torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None].bool()



class BertBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.model=AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
    self.lin=nn.Linear(v_size,768)
  def forward(self,  input_ids, attention_mask=None):
    return self.lin(self.model(input_ids, attention_mask=attention_mask)[0].squeeze())


class EncoderBlock(nn.Module):
    "Encoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha_bert=MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff  = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)
    
    def forward(self, x,bert_out, mask=None):
    # def forward(self, x, mask=None):
      att_self=self.mha(x,x)
      att_bertde=self.mha_bert(att_self,bert_out)
      return self.ff(att_bertde)
      # return self.ff(att_self)
      

class DecoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha2 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha_bert = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)
    
    def forward(self, x, enc,bert_out, mask_out=None):
    # def forward(self, x, enc, mask_out=None):
      # print (x.shape)
      att_self=self.mha1(x,x,mask_out)
      # print('att_self' ,att_self.shape)
      att_endec=self.mha2(att_self, enc)
      # print (att_endec.shape)
      # att_bertde=self.mha_bert(att_self,bert_out)
      # att_net=(att_endec+att_bertde)/2
      att_net=att_endec
      return self.ff(att_net)
      # return self.ff(att_endec)


      
      

class Transformer(Module):
    def __init__(self,v_size, n_layers=6, n_heads=8, d_model=768, d_head=32, 
                 d_inner=768, p=0.1, bias=True, scale=True, double_drop=True, pad_idx=1):
        # with torch.no_grad():
        self.emb=TransformerEmbedding(inp_emb)
        
        args = (n_heads, d_model, d_head, d_inner, p, bias, scale, double_drop)
        self.encoder = nn.ModuleList([EncoderBlock(*args) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(*args) for _ in range(n_layers)])
        self.bert=BertBlock()
        self.out = nn.Linear(d_model, v_size)
        self.pad_idx = pad_idx
        self.new_tensor = torch.LongTensor
        
        
    def forward(self, inp, out=None,gs=False):
        if gs:
            return self.greedy_search(inp.unsqueeze(dim=0))
        else:
            mask_out = get_output_mask(out, self.pad_idx)
            # with torch.no_grad():
            enc,out = self.emb(inp),self.emb(out)
            bert_out=self.bert(inp)
            # print("the shapes are ",bert_out.shape,enc.shape)
            enc = compose(self.encoder)(enc,bert_out)
            # enc = compose(self.encoder)(enc)
            

            out = compose(self.decoder)(out, enc,bert_out, mask_out)
            # out = compose(self.decoder)(out, enc, mask_out)
            return self.out(out)
    def greedy_search(self, inp, max_len=20):
        # print ("the input is ",inp)
        
        enc=self.emb(inp)
        # print("th emb is ",enc)
        # print("the shape of input is ",enc.shape)
        
        bert_out=self.bert(inp).unsqueeze(dim=0)
        # print("the shapes are ",bert_out.shape,enc.shape)
        enc = compose(self.encoder)(enc,bert_out)
        
        array=[0]
        # out=torch.tensor([0])
        for i in range(max_len):
            out=self.new_tensor(array).unsqueeze(dim=0).to(device)
            out=self.emb(out)
            
            out=compose(self.decoder)(out, enc,bert_out)
            out=self.out(out)
            
            out=out.detach()
            # print("the shape of the out is ",out.shape)
            # print("-1",out[:,-1,:].squeeze().shape)
            token=torch.argmax(out[:,-1,:].squeeze())
            array.append(token.item())
        # print(tokenizer.decode(array))
        # print(tokenizer.decode(inp))
        return array[1:]

# inp_emb(torch.tensor([1,2,3])).shape



torch.cuda.empty_cache()

import gc
# del variables
gc.collect()



model=Transformer(d_model=768, d_inner=768,v_size=v_size).to(device)

"""cuda"""
torch.cuda.empty_cache()
r = torch.cuda.memory_reserved()
a = torch.cuda.memory_allocated()
f = r-a
print (r,a,f)


optimizer=torch.optim.Adam(params=model.parameters(), lr=5e-4)

opt=optimizer

len(list(model.parameters()))





## **WandB**





"""cuda"""
torch.cuda.empty_cache()
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a
print (r,a,f)


# learn.destroy()

def evaluate(inputs, outputs, targets):
  print(outputs)
  print("targets ",targets)
  bleu_tup = compute_bleu([[x] for x in outputs], targets, smooth=False)
  bleu = bleu_tup[0]
  print(bleu_tup)
  return bleu
epoch=20

c_epoch=0
while True:
    c_epoch+=1

    train_loss=valid_loss=0.
    j=0
    for i,a in enumerate(data.train_dl):
      j+=1
      iter = i

      source_ids, target_ids,output_ids = a["source_ids"],a['target_ids'],a['output_ids']
      # print ((self.target_ids))
      # self('begin_batch')

      pred = model(inp=source_ids.to(device),out=output_ids.to(device))
      # self.pred.append(pred)
      pred=pred
      # self('after_pred')
      loss_func=CrossEntropyFlat()
      loss = loss_func(pred, target_ids.to(device))
      
      train_loss+=loss
      
      # self.train_loss+=self.loss if self.in_train else self.valid_loss+=self.loss
      # self('after_loss')
      # if not self.in_train: return
      loss.backward()                         
      opt.step()                           
      opt.zero_grad()
     
    print("train_loss ",train_loss/j)
      # print("valid_loss ")
    j=0
    torch.cuda.empty_cache()
    if c_epoch>0:
      model.eval()
      for i,a in enumerate(data.valid_dl):
        j+=1
        iter = i
        source_ids, target_ids,output_ids = a["source_ids"],a['target_ids'],a['output_ids']
        # print ((self.target_ids))
        # self('begin_batch')

        pred = model(inp=source_ids.to(device),out=output_ids.to(device))
        # self.pred.append(pred)
        pred=pred
        # self('after_pred')
        loss_func=CrossEntropyFlat()
        loss = loss_func(pred, target_ids.to(device))
        
        valid_loss+=loss
      
      print("valid_loss ",valid_loss/j)
      model.train()
    if c_epoch>16:
        model.eval()
        i=0
        outputs=[]
        inputs=[]
        targets=[]
        for i,a in enumerate(data.valid_dl):
            j+=1
            iter = i
            
            source_ids, target_ids,output_ids = a["source_ids"],a['target_ids'],a['output_ids']
            inp=[e for e in source_ids]
            targ=[e for e in target_ids]

            for m,j in enumerate(inp):
                pred=model(j.to(device),gs=True)

                l=['<s>','</s>','<pad>','�']
                # y=tokenizer.decode(pred)
                y=[tokenizer.decode([id]) for id in pred if tokenizer.decode([id]) not in l]
                # print("the type of y is ",y)
                # y=[a for a in y if a not in l ]
                # print("the after y is ",y)
                # z=tokenizer.decode(targ[m].tolist())
                z=[tokenizer.decode([id]) for id in targ[m].tolist() if tokenizer.decode([id]) not in l]
                # print("the z is ",z)
                # z=[a for a in z if a not in l ]
                # print("the after z is ",z)
                # print('y is ',y)
                # print('z is ',z)
                # # outputs.append(' '.join(y))
                # targets.append(' '.join(z))
                outputs.append(y)
                targets.append(z)



            # pred = model(inp=source_ids.to(device),gs=True)
        model.train()
        print("the bleu score is ",evaluate(None,outputs,targets))
    if (c_epoch>=epoch):
      print("end of training")
      # model.save('last_model.bin')
      break


## **Inference**




