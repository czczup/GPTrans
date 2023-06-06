import torch
import torch.distributed as dist
from .dataset import GraphDataset
from .dataset import BatchedDataDataset


def build_loader(config):
    config.defrost()
    dataset = GraphDataset(dataset_spec=config.DATA.DATASET, dataset_source=config.DATA.SOURCE)
    config.freeze()
    
    dataset_train = BatchedDataDataset(dataset.dataset_train, config.DATA.TRAIN_MAX_NODES,
                                       config.DATA.MULTI_HOP_MAX_DIST, config.DATA.SPATIAL_POS_MAX)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()}"
          f"successfully build train dataset, with #samples: {len(dataset_train)}")

    dataset_val = BatchedDataDataset(dataset.dataset_val, config.DATA.INFER_MAX_NODES,
                                      config.DATA.MULTI_HOP_MAX_DIST, config.DATA.SPATIAL_POS_MAX)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()}"
          f"successfully build val dataset, with #samples: {len(dataset_val)}")

    dataset_test = BatchedDataDataset(dataset.dataset_test, config.DATA.INFER_MAX_NODES,
                                      config.DATA.MULTI_HOP_MAX_DIST, config.DATA.SPATIAL_POS_MAX)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()}"
          f"successfully build test dataset, with #samples: {len(dataset_test)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    if dataset_train is not None:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True)

    if dataset_val is not None:
        if config.TEST.SEQUENTIAL:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False)

    if dataset_test is not None:
        if config.TEST.SEQUENTIAL:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_test = torch.utils.data.distributed.DistributedSampler(
                dataset_test, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        collate_fn=dataset_train.collater,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True) if dataset_train is not None else None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        collate_fn=dataset_val.collater,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        collate_fn=dataset_test.collater,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_test is not None else None

    return dataset_train, dataset_val, dataset_test, data_loader_train, \
        data_loader_val, data_loader_test




