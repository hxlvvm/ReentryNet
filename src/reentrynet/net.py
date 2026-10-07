"""A small U-Net with three outputs: inducibility, re-entry map and S1 activation map."""
from __future__ import annotations

import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ReentryUNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.down = nn.ModuleList([block(4, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([block(2 * w[i], w[i]) for i in range(3)])
        self.maps = nn.Conv2d(w[0], 2, 1)                       # re-entry logit, activation time
        self.cls = nn.Sequential(nn.Linear(2 * (w[3] + w[0]), 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, c):
        # the protocol is not translation invariant (S1 from the left edge, S2 at mid-height), so give positions
        h, w = c.shape[-2:]
        yy = torch.linspace(-1, 1, h, device=c.device).view(1, h, 1).expand_as(c)
        xx = torch.linspace(-1, 1, w, device=c.device).view(1, 1, w).expand_as(c)
        x = torch.stack([c, (c == 0).float(), xx, yy], 1)
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else self.pool(x))
            skips.append(x)
        bottom = x
        for i in reversed(range(3)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        maps = self.maps(x)
        pooled = torch.cat([bottom.mean((2, 3)), bottom.amax((2, 3)), x.mean((2, 3)), x.amax((2, 3))], 1)
        return self.cls(pooled).squeeze(1), maps[:, 0], maps[:, 1]
