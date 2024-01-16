import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import remove_spectral_norm, spectral_norm


def get_modules_of_type(module, types=None):
    default_types = (nn.Conv1d,
                     nn.Conv2d,
                     nn.Conv3d,
                     nn.ConvTranspose1d,
                     nn.ConvTranspose2d,
                     nn.ConvTranspose3d)

    types = default_types if types is None else default_types + (types)

    layers = []

    for layer in module.children():
        if any([isinstance(layer, t) for t in types]):
            layers.append(layer)
        elif isinstance(layer, nn.Module):
            layers.extend(get_modules_of_type(layer, types))

    return layers


def recursive_spectral_norm(model, types=None):
    """
    Recursively traverse submodules in a module and apply spectral norm to
    all convolutional layers.
    """
    modules = get_modules_of_type(model, types)

    for m in modules:
        if not hasattr(m, '_has_spectral_norm'):
            spectral_norm(m)
        setattr(m, '_has_spectral_norm', True)


class ConvBlock(nn.Module):
    def __init__(self, ch1, ch2, s):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(ch1),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, stride=s, padding_mode='reflect')
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, init_ch, n_layers=4):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
        )

        self.encoder = nn.ModuleList()

        ch = init_ch
        for _ in range(n_layers):
            self.encoder += [ConvBlock(ch, ch * 2, s=2)]
            ch *= 2

        self.bottleneck = nn.Sequential(
            nn.InstanceNorm2d(ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.in_proj(x)

        hidden_states = [x]

        for layer in self.encoder:
            x = layer(x)
            hidden_states.append(x)

        hidden_states[-1] = self.bottleneck(hidden_states[-1])
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, out_ch, init_ch, n_layers=4):
        super().__init__()

        self.projs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        ch = (2 ** n_layers) * init_ch
        for _ in reversed(range(n_layers)):
            self.projs += [nn.Conv2d(ch, ch // 2, kernel_size=1, padding_mode='reflect')]
            self.decoder += [ConvBlock(ch, ch // 2, s=1)]
            ch //= 2

        self.final_NA = nn.Sequential(
            nn.InstanceNorm2d(ch),
            nn.ReLU(),
        )

        self.out_proj = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, stride=1, padding_mode='reflect')

    def forward(self, hidden_states):
        x = hidden_states.pop()

        for proj, layer in zip(self.projs, self.decoder):
            x = proj(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            h = hidden_states.pop()
            x = torch.concat([h, x], dim=1)
            x = layer(x)

        return F.tanh(self.out_proj(self.final_NA(x)))


class UNet(nn.Module):
    def __init__(self, in_ch=1, init_ch=32, n_layers=4):
        super().__init__()
        self.encoder = Encoder(in_ch, init_ch, n_layers=n_layers)
        self.decoder = Decoder(in_ch, init_ch, n_layers=n_layers)

        recursive_spectral_norm(self.encoder)
        recursive_spectral_norm(self.decoder)
        remove_spectral_norm(self.decoder.out_proj)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)
