import os 
import random 
import cv2 
import torch 


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, file_path, transforms=None, phase='train'):
        self.root = root_dir 
        self.transforms = transforms
        
        self.phase = phase
        
        self.file_list = self._init_file(file_path)
    
    @staticmethod
    def _init_file(file_path):
        with open(file_path, 'r') as f:
            file_list = [x.strip() for x in f.readlines()]
        random.shuffle(file_list)
        
        return file_list 
    
    def __getitem__(self, idx):
        img_path, mask_path = self.file_list[idx].split(',')
        
        img = cv2.imread(os.path.join(self.root, img_path))[:, :, ::-1]
        mask = cv2.imread(os.path.join(self.root, mask_path), 0) / 255

        img_info = {
            'img_size': mask.shape[:2],
            'img_name': mask_path.split('/')[-1]
        }

        if self.transforms:
            sample = self.transforms(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']

        if self.phase == 'test':

            return img, mask, img_info
        
        return img, mask 
    
    def __len__(self):
        return len(self.file_list)
