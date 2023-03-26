from torch.nn import functional as F
import timm
from torch import nn
from segmentation_models_pytorch.base import (
    SegmentationModel as BaseSegModel,
    SegmentationHead,
)

from .utils import get_decoder_cls

class TimmEncoder(nn.Module):
	def __init__(self, name, depth=5, **kwargs):
		super().__init__()
		self.model = timm.create_model(name, **kwargs)

		self.out_channels = [ kwargs['in_chans'] ] + self.model.feature_info.channels()
		self.output_stride = min(32, 2**depth)

	def forward(self, x):
		features = self.model(x)
		features = [ x ] + features
		return features


class SegmentationModel(BaseSegModel):
    def __init__(
		self,
		encoder_name: str = "resnet34",
		encoder_depth: int = 5,
		decoder_name: str = 'Unet',
		decoder_use_batchnorm: bool = True,
		decoder_channels: list = (256, 128, 64, 32, 16),
		scale_factors: list = None,
		in_channels: int = 3,
		classes: int = 1,
		pretrained: bool = True,
		**kwargs
        ):

        super().__init__()

        self.encoder = TimmEncoder(
            encoder_name,
            depth=encoder_depth,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
            out_indices=tuple(range(encoder_depth)),
            **kwargs
        )

        self.decoder = get_decoder_cls(decoder_name)(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            scale_factors = scale_factors
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )
        self.classification_head = None

        self.name = "{}-{}".format(decoder_name, encoder_name).lower()
        self.initialize()