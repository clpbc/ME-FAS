# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from .transforms import FasTransforms
from .datasets import FasDataset
from .batchsampler import BatchSchedulerSampler, SchedulerSampler

def BuildDataset(rootPath, datasetType, labelType, isTrain, transforms):
    datasets = FasDataset(rootPath, datasetType, labelType, isTrain, transforms)
    return datasets

def BuildLoader(cfg, isTrain = True, log = None):
    abbr2datasetType = {'o' : 'oulu', 'c' : 'casia', 'm' : 'msu', 'i' : 'replay', 'b': 'celeb'}

    transforms = FasTransforms(cfg, isTrain)

    if isTrain:
        if log:
            log.write('Source Domain\n', is_file = True)
        else:
            print('\nSource Domain')

        srcDomains = cfg['dataset']['source'].lower()
        datasetTypes = [abbr2datasetType[i] for i in srcDomains]

    else:
        if log:
            log.write('\nTarget Domain\n', is_file = True)
        else:
            print('\nTarget Domain')

        desDomains = cfg['dataset']['target'].lower()
        datasetTypes = [abbr2datasetType[i] for i in desDomains]

    datasets = []
    for datasetType in datasetTypes:
        datasets.append(BuildDataset(cfg['dataset']['root_path'], datasetType, 'real', isTrain, transforms))
        datasets.append(BuildDataset(cfg['dataset']['root_path'], datasetType, 'fake', isTrain, transforms))

        if log:
            log.write(f"{datasetType} [real:{len(datasets[-2])}  fake:{len(datasets[-1])}]\n", is_file = True)
        else: 
            print(f"{datasetType} [real:{len(datasets[-2])}  fake:{len(datasets[-1])}]")

    labelTypeSize = cfg['dataset']['size']   # sample size of each datasetType and labelType in each batch
    concatDataset = ConcatDataset(datasets)

    # batch_sampler = BatchSchedulerSampler(concat_dataset, cfg['dataset']['batch_size'], train)
    # dataloader = DataLoader(concatDataset, batch_sampler = batch_sampler, num_workers = cfg['dataset']['num_workers'])

    sampler = SchedulerSampler(datasets = concatDataset, batch_size = labelTypeSize) 
    dataloader = DataLoader(concatDataset, sampler = sampler, num_workers = cfg['dataset']['num_workers'], batch_size = labelTypeSize * len(concatDataset.datasets))

    return dataloader
