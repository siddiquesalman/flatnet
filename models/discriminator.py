from sacred import Experiment
from typing import TYPE_CHECKING

import torch
from torch import nn

from config import initialise

if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("Disc")
ex = initialise(ex)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.args = args

        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_channels=128, num_groups=args.num_groups),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_channels=128, num_groups=args.num_groups),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_channels=256, num_groups=args.num_groups),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, img):
        logit = self.disc(img).squeeze()
        return logit


@ex.automain
def main(_run):
    from utils.tupperware import tupperware

    args = tupperware(_run.config)
    from math import ceil

    batch = 2

    img_ll = [
        torch.randn(batch, 3, ceil(args.image_height / 4), ceil(args.image_width / 4)),
        torch.randn(batch, 3, ceil(args.image_height / 2), ceil(args.image_width / 2)),
        torch.randn(batch, 3, args.image_height, args.image_width),
    ]
    D = Discriminator(args)

    D(img_ll)
