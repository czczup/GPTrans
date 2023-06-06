# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
import torch.distributed as dist
from .pyg_dataset import PYGDataset


class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyZINC(ZINC):
    def __init__(self, subset=True, **kwargs):
        super(MyZINC, self).__init__(subset=subset, **kwargs)
        
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyTSP(GNNBenchmarkDataset):
    def __init__(self, **kwargs):
        super(MyTSP, self).__init__(name="TSP", **kwargs)
    
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyTSP, self).download()
        if dist.is_initialized():
            dist.barrier()
    
    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyTSP, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyPattern(GNNBenchmarkDataset):
    def __init__(self, **kwargs):
        super(MyPattern, self).__init__(name="PATTERN", **kwargs)
        
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPattern, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPattern, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyCluster(GNNBenchmarkDataset):
    def __init__(self, **kwargs):
        super(MyCluster, self).__init__(name="CLUSTER", **kwargs)
        self.url = 'https://github.com/czczup/storage/releases/download/v0.1.0/'
    
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyCluster, self).download()
        if dist.is_initialized():
            dist.barrier()
    
    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyCluster, self).process()
        if dist.is_initialized():
            dist.barrier()
            

class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = "data"
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
        elif name == "zinc-subset":
            inner_dataset = MyZINC(root=root, subset=True)
            train_set = MyZINC(root=root, split="train", subset=True)
            valid_set = MyZINC(root=root, split="val", subset=True)
            test_set = MyZINC(root=root, split="test", subset=True)
        elif name == "zinc-full":
            inner_dataset = MyZINC(root=root, subset=False)
            train_set = MyZINC(root=root, split="train", subset=False)
            valid_set = MyZINC(root=root, split="val", subset=False)
            test_set = MyZINC(root=root, split="test", subset=False)
        elif name == "pattern":
            inner_dataset = MyPattern(root=root)
            train_set = MyPattern(root=root, split="train")
            valid_set = MyPattern(root=root, split="val")
            test_set = MyPattern(root=root, split="test")
        elif name == "cluster":
            inner_dataset = MyCluster(root=root)
            train_set = MyCluster(root=root, split="train")
            valid_set = MyCluster(root=root, split="val")
            test_set = MyCluster(root=root, split="test")
        elif name == "tsp":
            inner_dataset = MyTSP(root=root)
            train_set = MyTSP(root=root, split="train")
            valid_set = MyTSP(root=root, split="val")
            test_set = MyTSP(root=root, split="test")
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=root, name=nm)
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return PYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            return (
                None
                if inner_dataset is None
                else PYGDataset(inner_dataset, seed)
            )
