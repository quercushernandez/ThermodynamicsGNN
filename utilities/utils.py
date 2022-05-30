"""utils.py"""

import os
import numpy as np
import argparse

def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_log(args, log, name):
    file_name = name + '_log_' + args.sys_name + '.txt'
    save_dir = os.path.join(args.output_dir, file_name)

    epoch = log['epoch']
    loss_z = log['loss_z']
    loss_deg_E = log['loss_deg_E']
    loss_deg_S = log['loss_deg_S']

    f = open(save_dir, "w")
    f.write('epoch loss_z loss_deg_E loss_deg_S\n')
    for idx in range(len(epoch)):
        f.write(str(epoch[idx]) + " " + str(loss_z[idx]) + " ")
        f.write(str(loss_deg_E[idx]) + " " + str(loss_deg_S[idx]) + "\n")
    f.close()


def print_error(error):

    print('State Variable (L2 relative error)')
    for key in error.keys():
        e = error[key]
        error_mean = sum(e) / len(e)
        print('  ' + key + ' = {:1.2e}'.format(error_mean))


