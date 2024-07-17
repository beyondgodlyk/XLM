import torch
import torch.nn as nn
import torch.nn.functional as F
from xlm.model.transformer import gelu

class Classifier(nn.Module):
    def __init__(self, emb_dim, kernel_sizes, dropout, fc_sizes, num_filters, max_seq_len = 30):
        super(Classifier, self).__init__()

        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

        self.fcs = nn.ModuleList([
            nn.Linear(fc_sizes[i], fc_sizes[i + 1]) for i in range(len(fc_sizes) - 1)
        ])
        self.act = gelu

        self.init_weights()

    def init_weights(self):
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)
        for fc in self.fcs:
            nn.init.xavier_normal_(fc.weight)
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, latent):
        # TODO : Implement Spatial Dropout
        # latent (the 0th and 1st dimensions are already swapped in trainer): (batch_size, max_seq_len, emb_dim)
        assert latent.size(1) == self.max_seq_len + 2
        assert latent.size(2) == self.emb_dim

        latent = latent.transpose(1, 2).contiguous() # (batch_size, emb_dim, max_seq_len)

        conv_outs = []
        for conv in self.convs:
            conv_out = conv(latent)
            conv_out = self.act(conv_out)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outs.append(conv_out)

        conv_outs = torch.cat(conv_outs, 1) # (batch_size, num_filters * len(kernel_sizes))
        conv_outs = self.dropout(conv_outs)
        for fc in self.fcs:
            conv_outs = self.act(fc(conv_outs))
            conv_outs = self.dropout(conv_outs)
        return conv_outs