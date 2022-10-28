import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from fedavg import fedavg_main
from options import args_parser
from utils.helper import Helper as helper

if __name__ == '__main__':
    args           = args_parser()
    wandb = helper.init_wandb(args)
    args.save_dirs = helper.get_save_dirs(args.exp_dir, args.exp_name)
    helper.set_seed(args.seed)
    helper.backup_codes('../src', args.save_dirs['codes'], save_types=['.py', '.txt', '.sh', '.out'])
    logger         = SummaryWriter(args.save_dirs['logs'])
    
    if 'fedavg' in args.exp_name.lower():
        fedavg_main(args, logger, wandb)
    
    