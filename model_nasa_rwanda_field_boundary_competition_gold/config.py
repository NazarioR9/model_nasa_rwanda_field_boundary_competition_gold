import os

class CFG:
  month = ['2021_03', '2021_04', '2021_08', '2021_10', '2021_11', '2021_12']
  unet_chan = 3

  arch = 'regnetz_040h'
  lr = 2e-3
  wd = 1e-6
  epochs = 60
  warmup = 0.
  bs = 8
  ebs = 16

  img_size = 768
  threshold = 0.5

  fp16 = True
  workers = os.cpu_count()

  extra_params = {}