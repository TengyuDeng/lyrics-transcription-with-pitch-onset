from .datasets import *
from .collate_fns import get_collate_fn
from .transforms import get_transform
from .batch_samplers import MyBatchSampler
import os

num_workers = 8

def get_dataloaders(configs):
    
    harmonics_shift = configs['harmonics_shift'] if 'harmonics_shift' in configs else False
    transform = get_transform(configs['feature_type'], configs['transform_params'], harmonics_shift=harmonics_shift)
    
    train_dataloaders = {}
    val_dataloaders = {}
    test_dataloaders = {}
    for dataloader_name in configs['train']:
        dataloader = get_dataloader(configs['train'][dataloader_name], transform, train=True)
        train_dataloaders[dataloader_name] = dataloader
    for dataloader_name in configs['val']:
        dataloader = get_dataloader(configs['val'][dataloader_name], transform, train=False)
        val_dataloaders[dataloader_name] = dataloader
    for dataloader_name in configs['test']:
        dataloader = get_dataloader(configs['test'][dataloader_name], transform, train=False)
        test_dataloaders[dataloader_name] = dataloader

    return train_dataloaders, val_dataloaders, test_dataloaders

def get_dataloader(configs, transform, train):
    
    dataset_type = configs["type"]
    collate_fn = get_collate_fn(configs["collate_fn"])
    batch_size = configs["batch_size"]

    if dataset_type == "mix":
        subdatasets = []
        for subdataset_type in configs['sub_types']:
            subdataset = get_dataset(subdataset_type, configs[subdataset_type], transform, train)
            subdatasets.append(subdataset)
        dataset = torch.utils.data.ConcatDataset(subdatasets)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_sampler=MyBatchSampler(dataset.cumulative_sizes, batch_size, configs['percentage']),
            pin_memory=True,
        )

    else:
        dataset = get_dataset(dataset_type, configs, transform, train)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            )

    return dataloader

def get_dataset(dataset_type, configs, transform, train):

    if dataset_type != "LibriSpeech":
        dataset = MyDataset(
            name=dataset_type,
            transform=transform,
            **configs
            )

    else:
        dataset = MyLibriSpeechDataset(
            root="/n/work1/deng/data/",
            transform=transform,
            **configs
        )
    return dataset