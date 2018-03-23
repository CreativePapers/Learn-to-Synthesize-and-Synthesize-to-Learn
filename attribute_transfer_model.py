import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class _Residual_Block(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(_Residual_Block, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Encoder(nn.Module):
    """
    Generator (convolutional encoder) to encode input images (image+ landmark heatmap)
    """
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Encoder, self).__init__()

        net = []
        net.append(nn.Conv2d(6, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        net.append(nn.InstanceNorm2d(conv_dim, affine=True))
        net.append(nn.ReLU(inplace=True))

        # Down sampling
        channel_dim = conv_dim
        for i in range(4):
            net.append(nn.Conv2d(channel_dim, channel_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            net.append(nn.InstanceNorm2d(channel_dim * 2, affine=True))
            net.append(nn.ReLU(inplace=True))
            channel_dim = channel_dim * 2

        # Residual blocks
        for i in range(repeat_num):
            net.append(_Residual_Block(dim_in=channel_dim, dim_out=channel_dim))

        self.main = nn.Sequential(*net)


    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    """
    Generator (convolutional decoder) takes attributes and latent features and synthesize new images conditioned on the attributes of interest
    """
    def __init__(self, conv_dim=64):
        super(Decoder, self).__init__()
        # seven attributes (expression) categories
        channel_dim = 16*conv_dim+7
        net = []

        for i in range(4):

            up_scale=2
            net.append(nn.Conv2d(channel_dim, channel_dim//2 * up_scale ** 2, kernel_size=3, padding=1))
            net.append(nn.PixelShuffle(up_scale))

            net.append(nn.InstanceNorm2d(channel_dim//2, affine=True))
            net.append(nn.ReLU(inplace=True))
            channel_dim = channel_dim // 2


        self.main = nn.Sequential(*net)
        self.conv1 = nn.Conv2d(channel_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2 = nn.Conv2d(channel_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.m=nn.Tanh()


    def forward(self, input, label):
        label = label.unsqueeze(2).unsqueeze(3)
        label = label.expand(label.size(0), label.size(1), input.size(2), input.size(3))
        x = torch.cat([input, label], 1)
        h = self.main(x)
        h_1=self.conv1(h)
        h_2=self.conv2(h)
        out_image=self.m(h_1)
        out_landmark = self.m(h_2)
        return out_image, out_landmark


class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self, image_size=128, first_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()

        net = []
        net.append(nn.Conv2d(6, first_dim, kernel_size=4, stride=2, padding=1))
        net.append(nn.LeakyReLU(0.01, inplace=True))

        channel_dim = first_dim
        for i in range(1, repeat_num):
            net.append(nn.Conv2d(channel_dim, channel_dim*2, kernel_size=4, stride=2, padding=1))
            net.append(nn.LeakyReLU(0.01, inplace=True))
            channel_dim = channel_dim * 2

        self.main = nn.Sequential(*net)
        self.conv1 = nn.Conv2d(channel_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # there are seven expression classes
        self.fc = nn.Linear(channel_dim * 2 * 2, 7)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        h_cls = h.view(-1, num_flat_features(h))
        return out_real.squeeze(), self.fc(h_cls)

