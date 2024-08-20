import random, math
from torch.utils.data.sampler import Sampler, RandomSampler


class BatchSchedulerSampler(Sampler):
    """
    对所有子数据集进行合并, 保持子数据集内部样本顺序不变, 每个小批量包含来自不同子数据集的样本
    """
    def __init__(self, dataset, batch_size, train):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.train = train
        
        # 计算所有子数据集的总长度
        self.total_size = sum(len(cur_dataset) for cur_dataset in dataset.datasets)

    def __len__(self):
        return self.total_size // self.batch_size

    def __iter__(self):
        all_indices = []
        push_index_val = 0
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            cur_indices = list(range(push_index_val, push_index_val + len(cur_dataset)))
            all_indices.extend(cur_indices)
            push_index_val += len(cur_dataset)

        if self.train:  # 如果是训练模式，打乱所有索引
            random.shuffle(all_indices)

        # 将索引分成指定大小的批次
        batches = [all_indices[i: i + self.batch_size] for i in range(0, len(all_indices), self.batch_size)]
        
        # 注意：不需要再次展平索引列表，因为我们已经正确地将它们分组为批次
        return iter(batches)


class SchedulerSampler(Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        self.number_of_datasets = len(datasets.datasets)
        self.largest_dataset_size = max([len(cur_dataset.imgPaths) for cur_dataset in datasets.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.datasets.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.datasets.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)

            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()

            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.datasets.cumulative_sizes[:-1]

        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)