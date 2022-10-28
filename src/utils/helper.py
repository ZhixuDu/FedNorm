import os, gzip, shutil
import torch
import torchvision
import torch.nn as nn
import numpy as np
import random, math
import pandas
import csv
import scipy.misc
from concurrent.futures.thread import ThreadPoolExecutor
# import imageio
from scipy.stats import truncnorm

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
 
class Helper:
  #All directories are end with /
  
  @staticmethod
  def get_accuracy(output, target, label=None, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if label is not None:
        correct    = torch.mul(correct, torch.eq(target.view(1, -1), label).expand_as(pred))
        batch_size = torch.eq(target.view(1, -1), label).sum()

    res = []
    for k in topk: 
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append((correct_k.item(), batch_size))
    return res

  @staticmethod
  def get_precision_recall(output, target, label, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = torch.mul(correct, torch.eq(target.view(1, -1), label).expand_as(pred))
    total_relevant = torch.eq(target.view(1, -1).expand_as(pred), label)
    total_retrived = torch.eq(pred, label)
    
    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      total_retrived = total_retrived[:k].view(-1).float().sum(0)
      total_relevant = total_relevant[:k].view(-1).float().sum(0)
      res.append((correct_k, total_retrived, total_relevant))
    return res
  
  @staticmethod
  def pairwise_L2(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
  
  @staticmethod
  def network_norm(Module):
      norm = 0.0
      counter = 0.0
      for name, param in Module.named_parameters():
          if 'weight' in name:
              counter += 1
              norm += param.clone().detach().norm()/torch.prod(torch.tensor(list(param.shape)))
          elif 'bias' in name:
              counter += 1
              norm += param.clone().detach().norm()/torch.prod(torch.tensor(list(param.shape)))
      return (norm/counter).item()
  
  @staticmethod
      
  ###======================== Systems ======================== ####
  @staticmethod
  def multithread(max_workers, func, *args):  
      with ThreadPoolExecutor(max_workers=20) as executor:
          func(args)
  '''
  with ThreadPoolExecutor(max_workers=20) as executor:
    for i, idx in enumerate(idxs_users):
        executor.submit(local_update_for_multithread, rgs, train_dataset, user_groups_train, logger, test_dataset, user_groups_test, global_model, global_stat, epoch, local_weights, local_losses, device, idx, i)
  '''
          
  ###======================== Utilities ====================== ####
  @staticmethod
  def add_util_parser(parser):
    #=== directories ===
    parser.add_argument('--exp_name', type=str, default='Test', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='../../experiments/', help='Locations to save different experimental runs.')
    
    #== plot figures ===
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    return parser
     
  @staticmethod
  def get_save_dirs(exp_dir, exp_name):
    exp_dir = os.path.join(exp_dir, exp_name)
    save_dirs = dict()
    save_dirs['codes']  = os.path.join(exp_dir, 'codes/')
    save_dirs['checkpoints']  = os.path.join(exp_dir, 'checkpoints/')
    save_dirs['logs']  = os.path.join(exp_dir, 'logs/')
    save_dirs['figures']  = os.path.join(exp_dir, 'figures/')
    save_dirs['results']  = os.path.join(exp_dir, 'results/')
    for name, _dir in save_dirs.items():
      if not os.path.isdir(_dir):
        print('Create {} directory: {}'.format(name, _dir))
        os.makedirs(_dir)
    return save_dirs

  @staticmethod  
  def backup_codes(src_d, tgt_d, save_types=['.py', '.txt', '.sh', '.out']):
    for root, dirs, files in os.walk(src_d):
      for filename in files:
        type_list = [filename.endswith(tp) for tp in save_types]
        if sum(type_list):
          file_path = os.path.join(root, filename)
          tgt_dir   = root.replace(src_d, tgt_d)
          if not os.path.isdir(tgt_dir):
            os.makedirs(tgt_dir)
          shutil.copyfile(os.path.join(root, filename), os.path.join(tgt_dir, filename))
      
  @staticmethod
  def try_make_dir(d):
    if not os.path.isdir(d):
      # os.mkdir(d)
      os.makedirs(d) # nested is allowed

  @staticmethod
  def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s
  
  @staticmethod
  def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #To let the cuDNN use the same convolution every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
  def adjust_learning_rate(optimizer, epoch, lr, decay_every=100):
      """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
      lr = lr * (0.5 ** (epoch // decay_every))
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  ###======================== Logs ======================== ####
  @staticmethod
  def init_wandb(args):
    '''
    wandb will automatically save the log
    
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
    wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
    
    '''
  
    import wandb

    args.exp_name = str(args.exp_name)\
      + "-ac_" + str(args.model) \
      + "-nm_" + str(args.norm) \
      + "-lep" + str(args.local_ep) \
      + "-seed"+ str(args.seed) \
      + "-cvt" + str(args.cvt_shift) \
      + "-iid" + str(args.iid) \
      + "-ueq" + str(args.unequal) \
      + "-lr"  + str(args.lr) \
      + "-bs"  + str(args.local_bs) \
      + "-gr"  + str(args.epochs) \
      + "-ds_" + str(args.dataset) \

    if not os.path.isdir(os.path.join(args.exp_dir, args.exp_name)):
        os.makedirs(os.path.join(args.exp_dir, args.exp_name))
        
    wandb.init(
      project="FedNorm",
      name=args.exp_name,
      id=args.exp_name,
      resume=None,
      dir=os.path.join(args.exp_dir, args.exp_name),
      config=args,
      job_type=str(args.model)+'_'+str(args.seed),
      group=str(args.model)+'_'+str(args.norm)
    )

    return wandb
  
  @staticmethod
  def log(logf, msg, mode='a', console_print=True):
    with open(logf, mode) as f:
        f.write(msg + '\n')
    if console_print:
        print(msg)
     
  @staticmethod
  def write_dict2csv(log_dir, write_dict, mode="a"):
    for key in write_dict.keys():
      with open(log_dir + key + '.csv', mode) as f:
        if isinstance(write_dict[key], str):
          f.write(write_dict[key])
        elif isinstance(write_dict[key], list):
          writer = csv.writer(f)
          writer.writerow(write_dict[key])
        else:
          raise ValueError("write_dict has wrong type")
  
  
 	###======================== Visualization ================= ###
  @staticmethod
  def save_images(samples, sample_dir, sample_name, offset=0, nrows=0):
    if nrows == 0:
      bs = samples.shape[0]
      nrows = int(bs**.5)
    if offset > 0:
      sample_name += '_' + str(offset)
    save_path = os.path.join(sample_dir, sample_name + '.png')
    torchvision.utils.save_image(samples.cpu(), save_path, nrow=nrows, normalize=True) 
 