from config import *
import numpy as np
from torch.utils.data import SubsetRandomSampler,DataLoader
from utils.imageNet_datasets import train_imageNet_datasets

def get_loader(model_name,
               attack_method,
               validation_split = 0.1,
               test_split = 0.1,
               test_batch_size = 2,
               stage = "train"):


    data_dir = adv_datasets + "/" +model_name + "/" + attack_method
    datasets = train_imageNet_datasets(data_dir)
    dataset_size = len(datasets)
    indices = list(range(dataset_size))
    split1 = int(np.floor(validation_split * dataset_size))
    split2 = int(np.floor((test_split+validation_split) * dataset_size))

    if shuffle_training_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices,test_indices = \
        indices[split2:], indices[:split1],indices[split1:split2]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(datasets, batch_size=train_batch_size,
                              sampler=train_sampler,num_workers=num_workers)
    validation_loader = DataLoader(datasets, batch_size=validation_batch_size,
                                   sampler=valid_sampler,num_workers=num_workers)
    test_loader = DataLoader(datasets,batch_size = test_batch_size,
                             sampler=test_sampler,num_workers=num_workers)

    return train_loader,validation_loader,test_loader
