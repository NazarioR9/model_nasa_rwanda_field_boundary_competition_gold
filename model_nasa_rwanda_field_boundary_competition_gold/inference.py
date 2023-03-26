import os
import numpy as np

import torch
from torch.utils.data import Dataset as TDataset, DataLoader

from accelerate import Accelerator

import albumentations as A

from model import get_model
from utils import read_tile_test

import warnings
warnings.simplefilter('ignore')

##--------- Augs  ------------

def get_common_transforms(size):
  return A.Compose([
      A.Resize(height=size, width=size, p=1)
  ])

##---- Dataset --------

class HarvestDataset(TDataset):
  def __init__(self, tids, months, size):
    super().__init__()

    self.tids = tids
    self.m = months
    self.size = size
    self.default_size = 256

    self.common_tfms = get_common_transforms(size)

  def __len__(self):
    return len(self.tids)
  
  def _get_ts_data(self, idx):
    field = []
    for m in self.m:
      field.append(
          read_tile_test(self.tids[idx], m)
      )

    return field

  def __getitem__(self, idx):
    tid = self.tids[idx]
    fields = self._get_ts_data(idx)
    
    if self.size != self.default_size:
      for i,_ in enumerate(self.m):
        transformed = self.common_tfms(image=fields[i])
        fields[i] = transformed['image']
    
    fields = np.array(fields)
    fields = torch.tensor(np.transpose(fields, (0, 3, 1, 2)), dtype=torch.float)

    return fields
  
## --------- Inference -----------

def resize_batch(images, masks, size):
    n = images.shape[0]
    tfms = get_common_transforms(size)
    
    resized_images = []
    resized_masks = []

    for i in range(n):
        transformed = tfms(image=images[i], mask=masks[i])
        
        resized_images.append( transformed['image'] )
        resized_masks.append( transformed['mask'] )

    return np.stack( resized_images ), np.stack( resized_masks )

def load_model(fold, args, path='../checkpoints/'):
  model = get_model(args)
  model.load_state_dict(torch.load(f'{path}best_{fold}.pt'))
  model.eval()
  
  return model

def predict(dataloader, model):    
    all_logits = []

    model.eval()
    with torch.no_grad():
        for x in dataloader:
            preds = model(x).squeeze()
            if len(preds.shape) == 2:
                preds = preds.unsqueeze(0)

            all_logits.append(
                preds.detach().cpu().sigmoid().numpy()
            )

    return np.vstack(all_logits)

def run_predict(fold, args, test_ids):
  ds = HarvestDataset(test_ids, args.month, args.img_size)
  dl = DataLoader(ds, args.bs, shuffle=False, num_workers=args.workers)

  model = load_model(fold, args)

  accelerator = Accelerator(fp16=args.fp16)
  model, dl = accelerator.prepare(model, dl)

  preds = predict(dl, model)

  return preds