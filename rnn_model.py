import torch.nn as nn
import torch
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # GRU: input_size=emb_dim, hidden_size=hid_dim
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # outputs: [batch_size, src_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, method='dot'):
        super().__init__()
        self.method = 'concat' if method in ('concat', 'additive', 'bahdanau') else method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        if self.method == 'general': # Multiplicative
            self.W = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        elif self.method == 'concat': # Additive / Bahdanau
            self.W = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        elif self.method == 'dot':
            assert enc_hid_dim == dec_hid_dim, "Dot attention requires equal hidden dims"

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden (Decoder state): [n_layers, batch_size, dec_hid_dim] -> 取最后一层: [batch_size, dec_hid_dim]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim]
        
        if isinstance(hidden, tuple): # LSTM case
            hidden = hidden
        src_len = encoder_outputs.shape[1]
        
        # 取最后一层hidden state
        hidden = hidden[-1] 
        
        # 维度扩展以进行广播: [batch_size, src_len, dec_hid_dim]
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        if self.method == 'dot':
            # [batch, 1, dim] * [batch, dim, src_len] -> [batch, 1, src_len]
            # 这里利用 bmm 进行批量矩阵乘法
            energy = torch.bmm(hidden.unsqueeze(1), encoder_outputs.transpose(1, 2)).squeeze(1)
            
        elif self.method == 'general':
            energy = self.W(encoder_outputs) # [batch, src_len, dec_hid]
            energy = torch.bmm(hidden.unsqueeze(1), energy.transpose(1, 2)).squeeze(1)
            
        elif self.method == 'concat':
            # Bahdanau: v * tanh(W * [h_enc; h_dec])
            combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            energy = torch.tanh(self.W(combined))
            energy = self.v(energy).squeeze(2) # [batch_size, src_len]
        
        # Masking: 将padding位置的attention score设为负无穷
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        return F.softmax(energy, dim=1)

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + dec_hid_dim + enc_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        # input: [batch_size] (单步输入)
        input = input.unsqueeze(1) # [batch, 1]
        embedded = self.dropout(self.embedding(input)) # [batch, 1, emb_dim]
        
        # 计算注意力权重 [batch, src_len]
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1) # [batch, 1, src_len]
        
        # 计算加权上下文向量 [batch, 1, enc_hid_dim]
        weighted = torch.bmm(a, encoder_outputs)
        
        # RNN输入: 拼接 Embedding 和 Context Vector
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        # output: [batch, 1, dec_hid_dim], hidden: [n_layers, batch, dec_hid_dim]
        output, hidden = self.rnn(rnn_input, hidden)
        
        # 预测层输入: 拼接 Embedding, Output, Context Vector
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src)
        
        # 创建Mask (非pad部分为1)
        mask = (src != 0).long()  # 假设 0 是 pad idx
        
        # 解码器第一个输入是 <sos>
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            
            # 决定下一步输入
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            
            # 如果使用Teacher Forcing，则下个输入是真实Target；否则是预测值
            input = trg[:, t] if teacher_force else top1
            
        return outputs