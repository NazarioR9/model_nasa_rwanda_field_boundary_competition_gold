import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.decoders.unet import decoder


class DecoderBlock(decoder.DecoderBlock):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
        scale_factor=2
    ):

      super().__init__(
          in_channels,
          skip_channels,
          out_channels,
          use_batchnorm,
          attention_type)

      self.scale_factor = scale_factor

    def forward(self, x, skip=None):
      x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")

      if skip is not None:
          x = torch.cat([x, skip], dim=1)
          x = self.attention1(x)
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.attention2(x)
      return x

class UnetDecoder(decoder.UnetDecoder):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
        scale_factors = None
    ):
        super().__init__(
            encoder_channels,
            decoder_channels,
            n_blocks,
            use_batchnorm,
            attention_type,
            center
        )

        if scale_factors is None:
            scale_factors = [2] * n_blocks

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        if n_blocks != len(scale_factors):
            raise ValueError(
                "Model depth is {}, but you provide `scale_factors` for {} blocks.".format(
                    n_blocks, len(scale_factors)
                )
            )
        
        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        scale_factors = scale_factors[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs, scale_factor=sc_factor)
            for in_ch, skip_ch, out_ch, sc_factor in zip(in_channels, skip_channels, out_channels, scale_factors)
        ]
        self.blocks = nn.ModuleList(blocks)