import sys
sys.path.append('staross')

import ditto

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, embedding_size, hid=None):
        super().__init__()

        hid = embedding_size if hid is None else hid

        self.attn = nn.Sequential(
            nn.Linear(embedding_size, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x, bs, mask=None):
        _, d, h, w = x.size()
        x = x.view(bs, -1, d, h, w)
        x = x.permute(0, 1, 3, 4, 2)
        
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -torch.inf

        attn_weights = attn_logits.softmax(dim=1)

        x = attn_weights * x
        x = x.sum(dim=1)

        x = x.permute(0, 3, 1, 2)

        return x

class TemporalUnet(ditto.SegmentationModel):
  def __init__(
      self, **kwargs
    ):

    super().__init__(**kwargs)

    self.attn = nn.ModuleList(
            [
                AttentionPooling(i)
                for i in self.encoder.out_channels[1:]
            ]
        )
    
  def forward(self, x):
    BS, SL, C, H, W = x.shape

    x = x.view(BS * SL, C, H, W)

    features = self.encoder(x)[1:]
    features = [None] + [
        attn(f, BS)
        for f, attn in zip(features, self.attn)
    ]

    decoder_output = self.decoder(*features)

    masks = self.segmentation_head(decoder_output)

    return masks
  
class HarvestModel(nn.Module):
  def __init__(self, args):
    super().__init__()

    model_args = {
        'encoder_name': args.arch,
        'decoder_name': 'UnetPlusPlus',
        'in_channels': args.unet_chan,
        'classes': 1,
        'pretrained': False
    }
    
    self.unet = TemporalUnet(**model_args, **args.extra_params)
  
  def forward(self, field):
    return self.unet(field)
  
def get_model(args):
  model = HarvestModel(args)
  return model