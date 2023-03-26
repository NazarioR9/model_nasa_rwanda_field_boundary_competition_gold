import os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import warnings
warnings.simplefilter('ignore')

from inference import run_predict, resize_batch
from config import CFG
from utils import clean_string, disable_warnings, seed_everything

SEED = 42

test_source_items = os.environ['INPUT_DATA']
output_path = os.environ['OUTPUT_DATA']

months = ['2021_03', '2021_04', '2021_08', '2021_10', '2021_11', '2021_12']
test_tiles = [clean_string(s) for s in os.listdir(test_source_items)]
test_ids = list(set([x.split('_')[0] for x in test_tiles]).difference(['']))
test_ids = [x for x in test_ids if "-" not in x]

if __name__ == '__main__':
  disable_warnings()
  seed_everything(SEED)

  args = CFG()

  n_split = 10
  use_folds = list(range(n_split))
  actual_split = len(use_folds)

  # run prediction
  all_preds = []
  for f in tqdm(use_folds):
    t_preds = run_predict(f, args, test_ids)
    all_preds.append(t_preds)

  # group predictions for tile_id
  pred_dict = {}
  for idx, tid in enumerate(tqdm(test_ids)):
    pred_dict[tid] = []
    for test_preds in all_preds:
      pred_dict[tid].append( test_preds[idx] )

  # submission
  best_thresold = 0.6895833333333333
  subs = []
  for i, tid in enumerate(tqdm(sorted(test_ids))):
      tid_preds = np.mean(pred_dict[tid], axis=0)
      tid_preds = (tid_preds >= best_thresold).astype(int).astype('float32')[None]

      if args.img_size != 256:
          _, tid_preds = resize_batch(tid_preds, tid_preds, 256)
      tid_preds = tid_preds.squeeze().astype('uint8')

      ftd = pd.DataFrame(tid_preds)
      ftd = ftd.unstack().reset_index().rename(columns={'level_0': 'row', 'level_1': 'column', 0: 'label'})
      ftd['tile_row_column'] = f'Tile{tid}_' + ftd['row'].astype(str) + '_' + ftd['column'].astype(str)
      ftd = ftd[['tile_row_column', 'label']]

      subs.append(ftd)
  subs = pd.concat(subs)
  subs.to_csv(f'output.csv', index=False) ###-----------
