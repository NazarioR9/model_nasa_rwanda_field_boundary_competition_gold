import os
import numpy as np
import rasterio as rio
import warnings, logging
import random
import torch

import warnings
warnings.simplefilter('ignore')

source = os.environ['INPUT_DATA']
dataset_id = os.environ['DATASET_ID']

def disable_warnings(strict=False):
	warnings.simplefilter('ignore')
	if strict:
		logging.disable(logging.WARNING)

def seed_everything(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def normalize(
    array: np.ndarray
):
    """ normalise image to give a meaningful output """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def clean_string(s: str) -> str:
    """
    extract the tile id and timestamp from a source image folder
    e.g extract 'ID_YYYY_MM' from 'nasa_rwanda_field_boundary_competition_source_train_ID_YYYY_MM'
    """
    s = s.replace(f"{dataset_id}_source_", '').split('_')[1:]
    return '_'.join(s)

def read_tile_test(tid, month):
  tile = f'{tid}_{month}'

  bd1 = rio.open(f"{source}/{dataset_id}_source_test_{tile}/B01.tif")
  bd1_array = bd1.read(1)
  bd2 = rio.open(f"{source}/{dataset_id}_source_test_{tile}/B02.tif")
  bd2_array = bd2.read(1)
  bd3 = rio.open(f"{source}/{dataset_id}_source_test_{tile}/B03.tif")
  bd3_array = bd3.read(1)

  b01_norm = normalize(bd1_array)
  b02_norm = normalize(bd2_array)
  b03_norm = normalize(bd3_array)

  field = np.dstack((b03_norm, b02_norm, b01_norm)).astype('float32')

  return field