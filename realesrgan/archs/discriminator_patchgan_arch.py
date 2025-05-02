import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class PatchGANDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(PatchGANDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(num_in_ch, num_feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf = num_feat
        for i in range(1, 3):  # Reduce the depth for a lightweight version
            layers += [
                nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            nf *= 2
        layers += [nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)