

from data import *

tokenizer=AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
tokenizer.add_special_tokens({'bos_token':'<s>','eos_token':'</s>','unk_token':'unk'})
model_hugging=AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
emb=model_hugging.get_input_embeddings()
v_size=tokenizer.vocab_size

"""transformer model adopted from fast.ai with some changes from https://openreview.net/forum?id=Hyl7ygStwB"""

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
        bs,seq_len = q.size(0),q.size(1)
        wq,wk,wv = map(lambda o: self.create_attn_mat(*o,bs),
                       zip((q,kv,kv),(self.q_wgt,self.k_wgt,self.v_wgt)))
        attn_score = wq @ wk.transpose(2,3)
        if self.scale: attn_score /= math.sqrt(self.d_head)
        if mask is not None: 
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = attn_prob @ wv
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)


def get_output_mask(inp):
    return torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None].bool()

class BertBlock(nn.Module):
  def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
    super().__init__()
    self.model=model_hugging
    self.lin=nn.Linear(v_size,d_model)
  def forward(self,  input_ids, attention_mask=None):
    return self.lin(self.model(input_ids, attention_mask=attention_mask)[0].squeeze())


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha_bert=MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff  = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)
    
    def forward(self, x,bert_out, mask=None):
      att_self=self.mha(x,x)
      att_bertde=self.mha_bert(att_self,bert_out)
      return self.ff(att_bertde)
      
class DecoderBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha2 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha_bert = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)
    
    def forward(self, x, enc, bert_out, mask_out=None):
        att_self=self.mha1(x,x,mask_out)
        att_endec=self.mha2(att_self, enc)
        att_bertde=self.mha_bert(att_self,bert_out)
        att_net=(att_endec+att_bertde)/2
        return self.ff(att_net)

class Model(nn.Module):
    def __init__(self, v_size=v_size, d_model=128, n_layers=4, n_heads=8, d_head=16, d_inner=128, p=0.1, bias=True):
        super(Model,self).__init__()
        self.embed=nn.Embedding(v_size, d_model)
        args = (n_heads, d_model, d_head, d_inner, p, bias)
        self.encoder = nn.ModuleList([EncoderBlock(*args) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(*args) for _ in range(n_layers)])
        # self.bert=nn.ModuleList([BertBlock(*args) for _ in range(n_layers)])
        self.bert=BertBlock(*args)
        self.out=nn.Linear(d_model, v_size)
    
    def forward(self, inp, out):
        mask = get_output_mask(out)
        bert_out=self.bert(inp)
        enc,out=self.embed(inp),self.embed(out)
        enc = compose(self.encoder)(enc,bert_out)
        out = compose(self.decoder)(out, enc,bert_out, mask)
        out=self.out(out)
        return out



