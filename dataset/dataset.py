from functools import lru_cache
from typing import Optional

import torch
import torch.utils.data as data

from .collator import collator
from .dgl_datasets import DGLDatasetLookupTable
from .pyg_datasets import PYGDatasetLookupTable
from .ogb_datasets import OGBDatasetLookupTable


class BatchedDataDataset(data.Dataset):
    def __init__(self, dataset, max_nodes=128, multi_hop_max_dist=5, spatial_pos_max=1024):
        super().__init__()
        self.dataset = dataset
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collator(
            samples,
            max_node=self.max_nodes,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


def pad_1d_unsqueeze_nan(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype).float()
        new_x[:] = float('nan')
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


class TargetDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        try:
            return torch.stack(samples, dim=0)
        except: # only for PATTERN and CLUSTER now
            max_node_num = max(sample.size(0) for sample in samples)
            samples = torch.cat([pad_1d_unsqueeze_nan(i, max_node_num) for i in samples])
            samples = samples - 1 # for PATTERN and CLUSTER here
            return samples


class GraphDataset:
    def __init__(self, dataset_spec: Optional[str] = None,
                 dataset_source: Optional[str] = None, seed: int = 0):
        super().__init__()
        if dataset_source == "dgl":
            self.dataset = DGLDatasetLookupTable.GetDGLDataset(dataset_spec, seed=seed)
        elif dataset_source == "pyg":
            self.dataset = PYGDatasetLookupTable.GetPYGDataset(dataset_spec, seed=seed)
        elif dataset_source == "ogb":
            self.dataset = OGBDatasetLookupTable.GetOGBDataset(dataset_spec, seed=seed)
        else:
            raise ValueError(f"Unknown dataset source {dataset_source}")
        self.setup()

    def setup(self):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data
