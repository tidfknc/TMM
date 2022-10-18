from dataset import *
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def get_train_valid_sampler(trainset):
    size = len(trainset) 
    idx = list(range(size)) 
    return SubsetRandomSampler(idx) 



def get_data_loaders(dataset_name = 'IEMOCAP', split='train', batch_size=32, num_workers=0, pin_memory=False, args = None):

    print('building datasets..')
    dataset = MyDataset(dataset_name, split, args)
    

    data_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler= get_train_valid_sampler(dataset) if split!='test' else None,
                              collate_fn=dataset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    return data_loader