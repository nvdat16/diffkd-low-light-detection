import torch
from torch import nn


class EMBlock(nn.Module):
    def __init__(self, in_channels: int = 32, mid_channels: int = 8, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=0),
                nn.Conv2d(mid_channels, mid_channels * 2, kernel_size, padding=0),
                nn.Conv2d(mid_channels * 2, mid_channels * 4, kernel_size, padding=0),
                nn.ConvTranspose2d(mid_channels * 4, mid_channels * 2, kernel_size, padding=0),
                nn.ConvTranspose2d(mid_channels * 2, mid_channels, kernel_size, padding=0),
                nn.ConvTranspose2d(mid_channels, 3, kernel_size, padding=0),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.convs[:-1]:
            x = torch.relu(layer(x))
        x = torch.relu(self.convs[-1](x))
        return x


class MBLLEN(nn.Module):
    def __init__(self, input_channels: int = 3, base_channels: int = 32, mid_channels: int = 8):
        super().__init__()
        self.fem = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.em0 = EMBlock(base_channels, mid_channels)
        self.fem_blocks = nn.ModuleList([nn.Conv2d(base_channels, base_channels, 3, padding=1) for _ in range(9)])
        self.em_blocks = nn.ModuleList([EMBlock(base_channels, mid_channels) for _ in range(9)])
        self.out_conv = nn.Conv2d(3 * 10, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fem = torch.relu(self.fem(x))
        em_com = self.em0(fem)

        for fem_block, em_block in zip(self.fem_blocks, self.em_blocks):
            fem = torch.relu(fem_block(fem))
            em_com = torch.cat([em_com, em_block(fem)], dim=1)

        return torch.relu(self.out_conv(em_com))
