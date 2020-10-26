import torch
from typing import TYPE_CHECKING
from sacred import Experiment

from functools import partial
import torch.nn.functional as F
import torch.nn as nn
from utils.ops import unpixel_shuffle
from config import initialise
from utils.tupperware import tupperware

if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("Unet")
ex = initialise(ex)


def group_norm(num_channels: int, args: "tupperware"):
    return nn.GroupNorm(num_channels=num_channels, num_groups=args.num_groups)


def convrelu(
    in_channels, out_channels, kernel=3, padding=1, stride=1, normaliser=nn.BatchNorm2d
):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, stride=stride),
        normaliser(out_channels),
        nn.ReLU(inplace=True),
    )


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, normaliser):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1
        )
        self.upsample = nn.PixelShuffle(up_scale)
        self.normaliser = normaliser(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.normaliser(x)
        x = self.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self, args: "tupperware", in_c: int = 4):
        super().__init__()

        self.args = args
        normaliser = partial(group_norm, args=args)

        self.layer0 = nn.Sequential(
            nn.Conv2d(
                in_c * args.pixelshuffle_ratio ** 2,
                32,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            normaliser(32),
            nn.ReLU(inplace=True),
        )  # size=(N, 32, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(32, 32, 1, 0, normaliser=normaliser)

        self.layer1 = nn.Sequential(
            convrelu(32, 32, 3, 1, 2, normaliser=normaliser),
            convrelu(32, 32, 3, 1, 1, normaliser=normaliser),
        )  # size=(N, 32, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(32, 32, 1, 0, normaliser=normaliser)

        self.layer2 = nn.Sequential(
            convrelu(32, 64, 3, 1, 2, normaliser=normaliser),
            convrelu(64, 64, 3, 1, 1, normaliser=normaliser),
        )  # size=(N, 64, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(64, 64, 1, 0, normaliser=normaliser)

        self.layer3 = nn.Sequential(
            convrelu(64, 64, 3, 1, 2, normaliser=normaliser),
            convrelu(64, 128, 3, 1, 1, normaliser=normaliser),
        )  # size=(N, 128, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(128, 128, 1, 0, normaliser=normaliser)

        self.layer4 = nn.Sequential(
            convrelu(128, 64, 3, 1, 2, normaliser=normaliser),
            convrelu(64, 64, 3, 1, 1, normaliser=normaliser),
        )  # size=(N, 64, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(64, 128, 1, 0, 1, normaliser=normaliser)

        self.conv_up3 = nn.Sequential(
            convrelu(128 + 128, 128, 3, 1, normaliser=normaliser),
            convrelu(128, 128, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up2 = nn.Sequential(
            convrelu(64 + 128, 64, 3, 1, normaliser=normaliser),
            convrelu(64, 64, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up1 = nn.Sequential(
            convrelu(32 + 64, 32, 3, 1, normaliser=normaliser),
            convrelu(32, 32, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up0 = nn.Sequential(
            convrelu(32 + 32, 32, 3, 1, normaliser=normaliser),
            convrelu(32, 32, 3, 1, 1, normaliser=normaliser),
        )

        self.upsample_8_16 = UpsampleBLock(128, 2, normaliser=normaliser)
        self.upsample_16_32 = UpsampleBLock(128, 2, normaliser=normaliser)
        self.upsample_32_64 = UpsampleBLock(64, 2, normaliser=normaliser)
        self.upsample_64_128 = UpsampleBLock(32, 2, normaliser=normaliser)
        self.upsample_128_256 = UpsampleBLock(32, 2, normaliser=normaliser)

        self.conv_original_size0 = convrelu(
            in_c * args.pixelshuffle_ratio ** 2, 32, 3, 1, normaliser=normaliser
        )
        self.conv_original_size1 = convrelu(32, 32, 3, 1, normaliser=normaliser)
        self.conv_original_size2 = convrelu(32 + 32, 32, 3, 1, normaliser=normaliser)

        self.get_image_256 = nn.Sequential(
            nn.Conv2d(32, 3 * args.pixelshuffle_ratio ** 2, 1), nn.ReLU()
        )
        self.conv_smooth = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, stride=1), nn.Tanh()
        )
        self.pixelshuffle_ratio = args.pixelshuffle_ratio

    def forward(self, img):
        x_original = self.conv_original_size0(img)
        x_original = self.conv_original_size1(x_original)  # 32 channels

        layer0 = self.layer0(img)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)  # 64 channels

        layer4 = self.layer4_1x1(layer4)
        layer3 = self.layer3_1x1(layer3)

        x = self.upsample_8_16(layer4)  # size=(N, 64, x.H/16, x.W/16)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)  # size=(N, 64, x.H/16, x.W/16)

        layer2 = self.layer2_1x1(layer2)
        x = self.upsample_16_32(x)  # size=(N, 64, x.H/8, x.W/8)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)  # size=(N, 128, x.H/8, x.W/8)

        layer1 = self.layer1_1x1(layer1)
        x = self.upsample_32_64(x)  # size=(N, 128, x.H/4, x.W/4)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)  # size=(N, 128, x.H/4, x.W/4)

        layer0 = self.layer0_1x1(layer0)
        x = self.upsample_64_128(x)  # size=(N, 128, x.H/2, x.W/2)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)  # size=(N, 64, x.H/2, x.W/2)

        x = self.upsample_128_256(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  # size=(N, 32, x.H, x.W)
        img_256 = self.get_image_256(x)
        img_shuffle = F.pixel_shuffle(img_256, self.pixelshuffle_ratio)
        img_shuffle = self.conv_smooth(img_shuffle)
        img_256_unshuffle = unpixel_shuffle(img_shuffle, self.pixelshuffle_ratio)

        return img_256_unshuffle


@ex.automain
def main(_run):
    from torchsummary import summary

    args = tupperware(_run.config)

    is_admm = "admm" in args.exp_name
    in_c = 3 if is_admm else 4

    model = Unet(args, in_c)
    model = model.to(args.device)

    summary(
        model,
        (in_c * args.pixelshuffle_ratio ** 2, args.image_height, args.image_width),
    )
