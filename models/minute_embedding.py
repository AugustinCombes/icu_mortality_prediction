import torch
import torch.nn as nn
import math

class PatientEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=180, tokenizer_codes=None, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        assert tokenizer_codes is not None

        assert d_model%2==0
        d_model = d_model//2

        #compute time encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.squeeze(pe)
        self.time_encoding = pe.to(device)

        #compute code embedding
        self.embedding = nn.Embedding(1+len(tokenizer_codes), d_model-1, padding_idx=0, device=device)

    def forward(self, codes, values, minutes):
        code_embedding = self.embedding(codes)
        value_embedding = torch.concat([code_embedding, values.unsqueeze(dim=-1)], dim=-1)
        time_embedding = self.time_encoding[minutes]
        
        x = torch.concat([value_embedding, time_embedding], dim=-1)
        # return self.dropout(x)
        return x