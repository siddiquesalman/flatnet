import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from sacred import Experiment
import numpy as np
import torch.nn as nn

from config import initialise
from utils.ops import roll_n
from utils.tupperware import tupperware

if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("FFT-Layer")
ex = initialise(ex)


def fft_conv2d(input, kernel):
    """
    Computes the convolution in the frequency domain given
    Expects input and kernel already in frequency domain!
    :param input: shape (B, Cin, H, W)
    :param kernel: shape (Cout, Cin, H, W)
    :param bias: shape of (B, Cout, H, W)
    :return:
    """
    input = torch.rfft(input, 2, onesided=False)
    kernel = torch.rfft(kernel, 2, onesided=False)

    # Compute the multiplication
    # (a+bj)*(c+dj) = (ac-bd)+(ad+bc)j
    real = input[..., 0] * kernel[..., 0] - input[..., 1] * kernel[..., 1]
    im = input[..., 0] * kernel[..., 1] + input[..., 1] * kernel[..., 0]

    # Stack both channels and sum-reduce the input channels dimension
    out = torch.stack([real, im], -1)

    out = torch.irfft(out, 2, onesided=False)
    return out


def get_wiener_matrix(psf, Gamma: int = 20000, centre_roll: bool = True):
    """
    Get PSF matrix
    :param psf:
    :param gamma_exp:
    :return:
    """

    # Gamma = 10 ** (-0.1 * gamma_exp)
    if centre_roll:
        for dim in range(2):
            psf = roll_n(psf, axis=dim, n=psf.size(dim) // 2)

    psf = psf.unsqueeze(0)

    H = torch.rfft(psf, 2, onesided=False)
    Habsq = H[:, :, :, 0].pow(2) + H[:, :, :, 1].pow(2)

    W_0 = (torch.div(H[:, :, :, 0], (Habsq + Gamma))).unsqueeze(-1)
    W_1 = (-torch.div(H[:, :, :, 1], (Habsq + Gamma))).unsqueeze(-1)
    W = torch.cat((W_0, W_1), -1)

    weiner_mat = torch.irfft(W, 2, onesided=False)

    return weiner_mat[0]


class FFTLayer(nn.Module):
    def __init__(self, args: "tupperware"):
        super().__init__()
        self.args = args

        # No grad if you're not training this layer
        requires_grad = not (args.fft_epochs == args.num_epochs)

        psf = torch.tensor(np.load(args.psf_mat)).float()

        psf_crop_top = args.psf_centre_x - args.psf_crop_size_x // 2
        psf_crop_bottom = args.psf_centre_x + args.psf_crop_size_x // 2
        psf_crop_left = args.psf_centre_y - args.psf_crop_size_y // 2
        psf_crop_right = args.psf_centre_y + args.psf_crop_size_y // 2

        psf_crop = psf[psf_crop_top:psf_crop_bottom, psf_crop_left:psf_crop_right]

        wiener_crop = get_wiener_matrix(
            psf_crop, Gamma=args.fft_gamma, centre_roll=False
        )

        self.wiener_crop = nn.Parameter(wiener_crop, requires_grad=requires_grad)

        self.normalizer = nn.Parameter(
            torch.tensor([1 / 0.0008]).reshape(1, 1, 1, 1), requires_grad=requires_grad
        )

        if self.args.use_mask:
            mask = torch.tensor(np.load(args.mask_path)).float()
            self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, img):
        pad_x = self.args.psf_height - self.args.psf_crop_size_x
        pad_y = self.args.psf_width - self.args.psf_crop_size_y

        # Pad to psf_height, psf_width
        self.fft_layer = 1 * self.wiener_crop
        self.fft_layer = F.pad(
            self.fft_layer, (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2)
        )

        # Centre roll
        for dim in range(2):
            self.fft_layer = roll_n(
                self.fft_layer, axis=dim, n=self.fft_layer.size(dim) // 2
            )

        # Make 1 x 1 x H x W
        self.fft_layer = self.fft_layer.unsqueeze(0).unsqueeze(0)

        # FFT Layer dims
        _, _, fft_h, fft_w = self.fft_layer.shape

        # Target image (eg: 384) dims
        img_h = self.args.image_height
        img_w = self.args.image_width

        # Convert to 0...1
        img = 0.5 * img + 0.5

        # Use mask
        if self.args.use_mask:
            img = img * self.mask

        # Do FFT convolve
        img = fft_conv2d(img, self.fft_layer) * self.normalizer

        # Centre Crop
        img = img[
            :,
            :,
            fft_h // 2 - img_h // 2 : fft_h // 2 + img_h // 2,
            fft_w // 2 - img_w // 2 : fft_w // 2 + img_w // 2,
        ]
        return img


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    model = FFTLayer(args).to(args.device)
    img = torch.rand(1, 4, 1280, 1408).to(args.device)

    model(img)
