from xlm.data.dataset import Dataset
import numpy as np
import torch

class TSTDataset(Dataset):
    def __init__(self, sent, pos, params, style):
        self.style = style
        self.max_len = params.max_len
        super().__init__(sent, pos, params)

    def batch_sentences(self, sentences):
        """
        This is created to forced the length of all sentences to be the same (32 after adding 2 EOS tokens)
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(self.max_len + 2, lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.eos_index
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths