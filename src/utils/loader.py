

from torchvision import datasets, transforms



class Loader:
  #Data loader for all kinds of data sets
  
  @staticmethod
  def load_data_set(args):
      if args.dataset.lower() == 'cifar10':
          return get_cifar10()
          
      elif args.dataset.lower() == 'cifar100':
          return get_cifar100()
  
  @staticmethod
  def get_cifar10(data_dir='../data/', augmentation=False):
      if augmentation:
         train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                    ])
                                                     
         test_transform  = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                    ])
      else:
         train_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                    ])
         
         test_transform  = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                    ])     
      
      train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
      test_dataset  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
      
      return train_dataset, test_dataset
  
  @staticmethod
  def get_cifar100(data_dir='../data/', augmentation=False):
    if augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(15),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343)),
                                              ])
        test_transform  = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                              ])
    else:
       train_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ])
       test_transform  = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                              ])                                              

    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)


if __name__ == '__main__':
  