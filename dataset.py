"""dataset.py"""

import os
import numpy as np
import itertools

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class CouetteDataset(Dataset):
    def __init__(self, sims, dset_dir):
        'Initialization'
        self.sims = sims
        self.dset_dir = dset_dir
        self.dims = {'z':5, 'q':2, 'q_0':0, 'n':2, 'f':0, 'g':2}
        self.dt = 1/150

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        Re, We = self.sims[index]    
        # Load data
        name = os.path.join(self.dset_dir, 'couette_Re_{:.1f}_We_{:.1f}.pt'.format(Re,We))
        data = torch.load(name)     

        return data

    def __len__(self):
        return len(self.sims)

    def get_stats(self):
        mean = 0  
        for sim in range(len(self.sims)):
            batch = Batch.from_data_list(self[sim])
            mean += batch.y.sum(0)
        mean = mean/len(self.sims)/len(batch.batch)
        var = 0
        for sim in range(len(self.sims)):
            batch = Batch.from_data_list(self[sim])
            var += ((batch.y - mean)**2).sum(0)
        var = var/len(self.sims)/len(batch.batch)
        std = var**0.5
        std[std==0] = 1

        return {'mean': mean, 'std': std}, None


class BeamDataset(Dataset):
    def __init__(self, sims, dset_dir):
        'Initialization'
        self.sims = sims
        self.dset_dir = dset_dir
        self.dims = {'z':12, 'q':3, 'q_0':0, 'n':2, 'f':3, 'g':0}
        self.dt = 1/20

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        load = self.sims[index][0]  
        # Load data
        name = os.path.join(self.dset_dir, 'beam_{}.pt'.format(load+1))
        data = torch.load(name)     

        return data

    def __len__(self):
        return len(self.sims)

    def get_stats(self):
        mean_1 = 0  
        mean_2 = 0 
        for sim in range(len(self.sims)):
            batch = Batch.from_data_list(self[sim])
            mean_1 += batch.y.sum(0)
            mean_2 += batch.f.sum(0)
        mean_1 = mean_1/len(self.sims)/len(batch.batch)
        mean_2 = mean_2/len(self.sims)/len(batch.batch)
        var_1 = 0
        var_2 = 0
        for sim in range(len(self.sims)):
            batch = Batch.from_data_list(self[sim])
            var_1 += ((batch.y - mean_1)**2).sum(0)
            var_2 += ((batch.f - mean_2)**2).sum(0)
        var_1 = var_1/len(self.sims)/len(batch.batch)
        var_2 = var_2/len(self.sims)/len(batch.batch)
        std_1 = var_1**0.5
        std_1[std_1==0] = 1
        std_2 = var_2**0.5
        std_2[std_2==0] = 1

        return {'mean': mean_1, 'std': std_1}, {'mean': mean_2, 'std': std_2}


class CylinderDataset(Dataset):
    def __init__(self, sims, dset_dir):
        'Initialization'
        self.sims = sims
        self.dset_dir = dset_dir
        self.dims = {'z':3, 'q':0, 'q_0':2, 'n':4, 'f':0, 'g':0}
        self.dt = 1/100

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        pos, v = self.sims[index]    
        # Load data
        name = os.path.join(self.dset_dir, 'cylinder_{}_v_{:.2f}.pt'.format(pos,v))
        data = torch.load(name)

        return data

    def __len__(self):
        return len(self.sims)

    def get_stats(self):
        'Computes the statistics of the dataset'
        # Mean
        mean_1 = 0  
        mean_2 = 0        
        for sim in range(len(self.sims)):
            batch = Batch.from_data_list(self[sim])
            mean_1 += batch.y.sum(0)
            mean_2 += batch.q_0.sum(0)
        mean_1 = mean_1/len(self.sims)/len(batch.batch)
        mean_2 = mean_2/len(self.sims)/len(batch.batch)
        # Variance
        var_1 = 0
        var_2 = 0
        for sim in range(len(self.sims)):
            batch = Batch.from_data_list(self[sim])
            var_1 += ((batch.y - mean_1)**2).sum(0)
            var_2 += ((batch.q_0 - mean_2)**2).sum(0)
        var_1 = var_1/len(self.sims)/len(batch.batch)
        var_2 = var_2/len(self.sims)/len(batch.batch)
        # Standard Deviation
        std_1 = var_1**0.5
        std_1[std_1==0] = 1
        std_2 = var_2**0.5
        std_2[std_2==0] = 1

        return {'mean': mean_1, 'std': std_1}, {'mean': mean_2, 'std': std_2}


def load_dataset(args):
    # Dataset directory path
    sys_name = args.sys_name
    dset_dir = os.path.join(args.dset_dir, 'database_' + sys_name)

    # Create Dataset instances
    if args.sys_name == 'couette':
        # Cases: Reynolds + Weisemberg
        Re = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        We = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
        train_sims, val_sims, test_sims = split_dataset(Re,We)
        
        train_set = CouetteDataset(train_sims, dset_dir)
        val_set = CouetteDataset(val_sims, dset_dir)
        test_set = CouetteDataset(test_sims, dset_dir)

    elif args.sys_name == 'beam':
        # Cases: Load
        load = list(range(52))
        train_sims, val_sims, test_sims = split_dataset(load)
        
        train_set = BeamDataset(train_sims, dset_dir)
        val_set = BeamDataset(val_sims, dset_dir)
        test_set = BeamDataset(test_sims, dset_dir)

    elif args.sys_name == 'cylinder':
        # Cases: Position + Velocity
        pos = [1, 2, 3, 4, 5, 6]
        v = [1, 1.25, 1.5, 1.75, 2]
        train_sims, val_sims, test_sims = split_dataset(pos,v)

        train_set = CylinderDataset(train_sims, dset_dir)
        val_set = CylinderDataset(val_sims, dset_dir)
        test_set = CylinderDataset(test_sims, dset_dir)

    return train_set, val_set, test_set


def split_dataset(*args):
    # Train, validation and test simulations
    indices = list(itertools.product(*args))
    N_sims = len(indices)

    # Random split
    np.random.shuffle(indices)

    train_sims = indices[:int(0.8*N_sims)]
    val_sims = indices[int(0.8*N_sims):int(0.9*N_sims)]
    test_sims = indices[int(0.9*N_sims):]

    return train_sims, val_sims, test_sims


if __name__ == '__main__':
    pass
