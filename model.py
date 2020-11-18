

from data import *

# tokenizer=AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
# tokenizer=AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
# tokenizer.add_special_tokens({'bos_token':'<s>','eos_token':'</s>','unk_token':'unk'})
# model_hugging=AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
# emb=model_hugging.get_input_embeddings()
v_size=tokenizer_code.get_vocab_size()
print("the vocab size is ",v_size)

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


def get_output_mask(inp, pad_idx=1):
    return torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None].bool()
#     return ((inp == pad_idx)[:,None,:,None].long() + torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None] != 0)

class EncoderBlock(nn.Module):
    "Encoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff  = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)
    
    def forward(self, x, mask=None): return self.ff(self.mha(x, x, mask=mask))

class DecoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha2 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)
    
    def forward(self, x, enc, mask_out=None): return self.ff(self.mha2(self.mha1(x, x, mask_out), enc))

class Model(nn.Module):
    def __init__(self, v_size=v_size, d_model=128, n_layers=4, n_heads=8, d_head=32, d_inner=256, p=0.1, bias=True):
        super(Model,self).__init__()
        self.embed_src=nn.Embedding(v_size, d_model)
        self.embed_code=nn.Embedding(v_size, d_model)
        args = (n_heads, d_model, d_head, d_inner, p, bias)
        self.encoder = nn.ModuleList([EncoderBlock(*args) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(*args) for _ in range(n_layers)])
        # self.bert=nn.ModuleList([BertBlock(*args) for _ in range(n_layers)])
        # self.bert=BertBlock(*args)
        self.out=nn.Linear(d_model, v_size)
        self.new_tensor = torch.LongTensor
    
    def forward(self, inp, out=None,gs=False):
        if gs:
            return self.greedy_search(inp)
        else:
            mask = get_output_mask(out)
            # bert_out=self.bert(inp)
            enc,out=self.embed_src(inp),self.embed_code(out)
            # print("the shapes are ",enc.shape, out.shape)
            enc = compose(self.encoder)(enc)

            # print("shape ",enc.shape)
            out = compose(self.decoder)(out, enc, mask)
            # print(enc.shape, out.shape)
            out=self.out(out)
            return out
    
    def greedy_search(self, inp, max_len=20):
        # print ("the input is ",inp)
        
        enc=self.embed_src(inp).unsqueeze(dim=0)
        # print("th emb is ",enc)
        # print("the shape of input is ",enc.shape)
        enc = compose(self.encoder)(enc)
        
        array=[0]
        # out=torch.tensor([0])
        for i in range(max_len):
            out=self.new_tensor(array).unsqueeze(dim=0).to(device)
            out=self.embed_code(out)
            
            out=compose(self.decoder)(out, enc)
            
            out=out.detach()
            # print("the shape of the out is ",out.shape)
            # print("-1",out[:,-1,:].squeeze().shape)
            token=torch.argmax(out[:,-1,:].squeeze())
            array.append(token.item())
        # print(tokenizer.decode(array))
        # print(tokenizer.decode(inp))
        return array






