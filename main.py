"""main.py"""

import argparse

import numpy as np
import torch

from solver import Solver
from utilities.utils import str2bool


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    solver = Solver(args)

    if args.train: solver.train_model()
    solver.test_model()
    if args.plot_sim: solver.plot_sim()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

    # Study Case
    parser.add_argument('--sys_name', default='couette', type=str, help='physic system name')
    parser.add_argument('--train', default=False, type=str2bool, help='train or test')
    parser.add_argument('--gpu', default=False, type=str2bool, help='GPU acceleration')

    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')

    # Net Parameters
    parser.add_argument('--n_hidden', default=2, type=int, help='number of hidden layers per MLP')
    parser.add_argument('--dim_hidden', default=10, type=int, help='dimension of hidden units')
    parser.add_argument('--passes', default=10, type=int, help='number of message passing')

    # Training Parameters
    parser.add_argument('--seed', default=1, type=int, help='random seed')   
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lambda_d', default=1e2, type=float, help='data loss weight')
    parser.add_argument('--noise_var', default=1e-2, type=float, help='training noise variance')
    parser.add_argument('--batch_size', default=2, type=int, help='training batch size')
    parser.add_argument('--max_epoch', default=6000, type=int, help='maximum training iterations')
    parser.add_argument('--miles', default=[2000, 4000], nargs='+', type=int, help='learning rate scheduler milestones')
    parser.add_argument('--gamma', default=1e-1, type=float, help='learning rate milestone decay')

    # Save and plot options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--plot_sim', default=True, type=str2bool, help='plot test simulation')

    args = parser.parse_args()

    main(args)
