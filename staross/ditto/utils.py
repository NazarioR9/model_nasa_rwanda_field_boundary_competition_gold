from .decoders import UnetDecoder, UnetPlusPlusDecoder

def get_decoder_cls(cls_name):
	if cls_name == 'UnetPlusPlus':
		return UnetPlusPlusDecoder

	return UnetDecoder