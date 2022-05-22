import torch 

from .dataset import Dataset
from .transforms import get_train_transform, get_test_transform 


def build_loader(cfg, phase='train'):

    if phase == 'train':

        train_cfg = cfg['TrainReader']
        train_dataset = Dataset(root_dir=train_cfg['root_dir'], file_path=train_cfg['file_path'], 
                                transforms=get_train_transform(cfg['input_size']))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'], shuffle=True, drop_last=False, pin_memory=True
        )
        return train_loader
    else:
        test_cfg = cfg['TestReader']
        test_dataset = Dataset(root_dir=test_cfg['root_dir'], file_path=test_cfg['file_path'],
                                transforms=get_test_transform(cfg['input_size']), phase='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_cfg['batch_size'], num_workers=test_cfg['num_workers'], shuffle=False, drop_last=False, pin_memory=True
        )
        return test_loader
