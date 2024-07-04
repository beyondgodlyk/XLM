import torch
import torch.nn as nn
import torch.nn.functional as F
from xlm.model.transformer import gelu

class Classifier(nn.Module):
    def __init__(self, emb_dim, kernel_sizes, max_seq_len = 30, fc_sizes=[100, 50, 1], num_filters=256, dropout=0.1):
        super(Classifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.convs = nn.ModuleList([
            nn.Conv2d(emb_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        fc_sizes = [num_filters * len(kernel_sizes), num_filters] + fc_sizes
        self.fcs = nn.ModuleList([
            nn.Linear(fc_sizes[i], fc_sizes[i + 1]) for i in range(len(fc_sizes) - 1)
        ])
        self.sigmoid = nn.Sigmoid()
        self.act = gelu

    def forward(self, con_latent):
        # TODO : Implement Spatial Dropout
        # con_latent (the 0th and 1st dimensions are already swapped): (batch_size, max_seq_len, max_seq_len * emb_dim)
        assert con_latent.size(1) == self.max_seq_len
        assert con_latent.size(2) == self.max_seq_len * con_latent.size(1)

        conv_outs = []
        for conv in self.convs:
            conv_out = conv(con_latent)
            conv_out = self.act(conv_out)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outs.append(conv_out)
        
        conv_outs = torch.cat(conv_outs, 1) # (batch_size, num_filters * len(kernel_sizes))
        conv_outs = self.dropout(conv_outs)
        for fc in self.fcs:
            conv_outs = self.act(fc(conv_outs))
            conv_outs = self.dropout(conv_outs)
        return self.sigmoid(conv_outs)