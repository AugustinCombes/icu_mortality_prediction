import torch
import torch.nn as nn
from models.minute_embedding import PatientEmbedding

class TransformerPredictor(nn.Module):

    def __init__(self, d_embedding, d_model, dropout=0.1, n_layers=2, tokenizer_codes=None, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        #patient embedding
        self.embedding = PatientEmbedding(d_model=d_embedding, tokenizer_codes=tokenizer_codes, device=device)

        #projection
        self.proj = nn.Linear(d_embedding, d_model).to(device)

        #transformer embedding
        layer = nn.TransformerEncoderLayer(d_model, 2, dim_feedforward=2*d_model, dropout=0.5, batch_first=True, device=device)
        self.transformer = nn.TransformerEncoder(layer, n_layers).to(device)

        #classification
        self.cls = nn.Linear(d_model, 1, bias=True).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, codes, values, minutes):
        x = self.embedding(codes, values, minutes)
        x = nn.ReLU()(x)
        
        x = self.proj(x)
        x = nn.ReLU()(x)

        x = self.transformer(x)
        x = x[:, -1, :]
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.cls(x)
        x = self.sigmoid(x)
        return x