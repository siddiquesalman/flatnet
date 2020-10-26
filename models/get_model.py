"""
Get model
"""
from models.fftlayer import FFTLayer
from models.admm.admm_first_block import ADMM_Net as ADMM

from models.unet_128 import Unet as Unet_128
from models.unet_64 import Unet as Unet_64
from models.unet_32 import Unet as Unet_32


from models.discriminator import Discriminator


def model(args):
    is_admm = "admm" in args.exp_name
    in_c = 3 if is_admm else 4
    Inversion = ADMM if is_admm else FFTLayer

    if args.model == "unet-128-pixelshuffle-invert":
        return Unet_128(args, in_c=in_c), Inversion(args), Discriminator(args)

    elif args.model == "unet-64-pixelshuffle-invert":
        return Unet_64(args, in_c=in_c), Inversion(args), Discriminator(args)

    elif args.model == "unet-32-pixelshuffle-invert":
        return Unet_32(args, in_c=in_c), Inversion(args), Discriminator(args)
