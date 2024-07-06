from xlm.data.dataset import Dataset
import numpy as np

class TSTDataset(Dataset):
    def __init__(self, sent, pos, params, style):
        self.style = style

        super().__init__(sent, pos, params)
