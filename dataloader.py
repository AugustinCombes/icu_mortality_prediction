import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd

class EHRDataset(Dataset):
    def __init__(self, path_documents="events.json", path_labels="targets.json", path_tokenizer="tokenizer.json", mode="train"):

        assert mode in ["train", "test"]
        
        # load data
        with open(path_documents) as f:
            self.data = json.load(f)
        with open(path_labels) as f:
            self.targets = json.load(f)

        # create tokenizer
        with open(path_tokenizer) as f:
            self.tokenizer = json.load(f)
        
        # restrict data to train or test set
        ref_k = list(self.data.keys()).copy()
        if mode == "train":
            for k in ref_k:
                if k.endswith("8") or k.endswith("9"):
                    del self.data[k]
                    del self.targets[k]
        else : #test mode
            for k in ref_k:
                if not k.endswith("8") or k.endswith("9"):
                    del self.data[k]
                    del self.targets[k]

        self.icu_stays_id = list(self.data.keys()) # for future indexing
                
        assert len(self.data) == len(self.targets)
        # print(f"{len(self.data)} available samples, {len(self.tokenizer)} tokens")

    def __len__(self):
        return len(self.icu_stays_id)

    def __getitem__(self, index):
        patient = self.data[self.icu_stays_id[index]]
        t_list, v_list = list(map(float, patient.keys()))+[179.], list(patient.values())+[[[1,0]]] #adds cls token at the end of the available data

        minutes = np.repeat(t_list, list(map(len, v_list)))
        minutes = torch.tensor(minutes).long()
        codes = torch.tensor([self.tokenizer.get(str(e[0]), len(self.tokenizer)) for v in v_list for e in v]).long()
        values = torch.tensor([e[1] for v in v_list for e in v])

        padding = (100 - minutes.size(0))
        minutes = torch.nn.functional.pad(minutes, (padding, 0))
        codes = torch.nn.functional.pad(codes, (padding, 0))
        values = torch.nn.functional.pad(values, (padding, 0))

        sample = {
            "codes": codes,
            "values": values,
            "minutes": minutes,
            "target": 1-self.targets[self.icu_stays_id[index]]
        }
        return sample