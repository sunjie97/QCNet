import os 
import yaml 
import argparse 
from tqdm import tqdm 
from PIL import Image 

import torch 
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from datasets import build_loader
from model import QCNet 
from utils import seed_everything, AverageMeter, BinaryDiceLoss
import warnings
warnings.filterwarnings('ignore')


class Solver:

    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(cfg['seed'])

    def train(self):
        num_epochs = int(self.cfg['num_epochs'])
        device = torch.device(self.cfg['device'])
        writer = SummaryWriter(self.cfg['save_dir'] + '/log')
        scaler = GradScaler()

        model = QCNet(backbone=self.cfg['backbone'], phase='train').to(device)
        model.train()
        loader = build_loader(self.cfg, phase='train')
        criterion = nn.BCEWithLogitsLoss().to(device)
        criterion_sal = BinaryDiceLoss().to(device)

        encoder_params = list(map(id, model.encoder.parameters()))
        decoder_params = filter(lambda p: id(p) not in encoder_params, model.parameters())
        params = [
            {'params': model.encoder.parameters(), 'lr': 5e-5},
            {'params': decoder_params, 'lr': 1e-4}
        ]
        optimizer = getattr(torch.optim, self.cfg['Optimizer']['type'])(params, weight_decay=1e-6)
        scheduler = getattr(torch.optim.lr_scheduler, self.cfg['Scheduler']['type'])(optimizer, **self.cfg['Scheduler']['args'])

        for epoch in range(num_epochs):

            train_loss = AverageMeter()
            for batch_idx, (imgs, targets) in enumerate(loader):
                imgs, targets = imgs.to(device).float(), targets.unsqueeze(1).to(device).float()
                
                with autocast():
                    preds, aux_preds = model(imgs)
                    loss = criterion(preds, targets)
                    for aux_pred in aux_preds:
                        loss += criterion_sal(aux_pred, targets) 
                    
                    scaler.scale(loss).backward()
                    
                    nn.utils.clip_grad_value_(model.parameters(), clip_value=float(self.cfg['clip']))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    train_loss.update(loss.item())

            scheduler.step()
            writer.add_scalar(f'train_loss', loss.item(), global_step=epoch)
            print(f'Training Epoch [{epoch}/{num_epochs}], training loss: {train_loss.average}')

    def test(self):
        device = torch.device(self.cfg['device'])
        loader = build_loader(self.cfg, phase='test')

        model = QCNet(backbone=self.cfg['backbone'], phase='test').to(device)
        state_dict = torch.load(self.cfg['infer_checkpoint'])
        for k in list(state_dict.keys()):
            if 'aux_seg_heads' in k:
                del state_dict[k]
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            for imgs, _, img_info in tqdm(loader, total=len(loader)):
                img_size = (int(img_info['img_size'][1]), int(img_info['img_size'][0]))
                # print('img_size: ', img_size)
                img_name = img_info['img_name'][0]
                
                imgs = imgs.to(device)

                out = model(imgs)

                out = out.detach().cpu().squeeze().sigmoid().numpy()
                out = Image.fromarray(out * 255).convert('L')
                out = out.resize(img_size, resample=1)

                infer_save_dir = self.cfg['infer_save_dir']
                if not os.path.exists(infer_save_dir):
                    os.makedirs(infer_save_dir) 

                out.save(f'{infer_save_dir}/{img_name}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--cfg', default='configs/composite.yaml')

    args = parser.parse_args()

    return args 

if __name__ == '__main__':

    args = get_args()
    solver = Solver(yaml.safe_load(open(args.cfg)))
    if args.phase == 'train':
        solver.train()
    else:
        solver.test()
    
