import torch
from torch import nn
from typing import List


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    @staticmethod
    def l2normalize(v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class MunitDiscriminatorConvBlock(nn.Module):
    """
    source: https://github.com/NVlabs/MUNIT/blob/a99d853ab3d5fda837395c821a7a65d46afdebb4/networks.py
    """

    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='sn', activation='relu', pad_type='zero'):
        super(MunitDiscriminatorConvBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad3d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad3d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad3d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.norm = None
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class MunitDiscriminator(nn.Module):
    """
    source: https://github.com/NVlabs/MUNIT/blob/a99d853ab3d5fda837395c821a7a65d46afdebb4/networks.py
    """

    def __init__(self,
                 input_dim=1,
                 n_layer=4,
                 dim=64,
                 norm='sn',
                 activ='leaky_relu',
                 num_scales=3,
                 pad_type='reflect',
                 ):
        self.input_dim = input_dim
        self.n_layer = n_layer
        self.dim = dim
        self.norm = norm
        self.activ = activ
        self.num_scales = num_scales
        self.pad_type = pad_type

        super(MunitDiscriminator, self).__init__()
        self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [MunitDiscriminatorConvBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ,
                                              pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [MunitDiscriminatorConvBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ,
                                                  pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv3d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


# Loss

@torch.jit.script
def fuse_math_min_mean_pos(x):
    r"""
    https://github.com/NVlabs/imaginaire/blob/c6f74845c699c58975fd12b778c375b72eb00e8d/imaginaire/losses/gan.py
    Fuse operation min mean for hinge loss computation of positive
    samples"""
    minval = torch.min(x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


@torch.jit.script
def fuse_math_min_mean_neg(x):
    r"""
    https://github.com/NVlabs/imaginaire/blob/c6f74845c699c58975fd12b778c375b72eb00e8d/imaginaire/losses/gan.py
    Fuse operation min mean for hinge loss computation of negative
    samples"""
    minval = torch.min(-x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


def disc_hinge_loss(disc_outputs: List[torch.Tensor], disc_update: bool, real: bool) -> torch.Tensor:
    """
    https://github.com/NVlabs/imaginaire/blob/c6f74845c699c58975fd12b778c375b72eb00e8d/imaginaire/losses/gan.py
    @param disc_outputs: the output from the discriminator
    @param disc_update: is this used to update the discriminator in the next step?
    @param real: is it a real sample?
    @return:
    """
    if not disc_update:
        assert real, \
            "The target should be real when updating the generator."

    losses = []
    for disc_output in disc_outputs:
        if disc_update:
            if real:
                loss = fuse_math_min_mean_pos(disc_output)
            else:
                loss = fuse_math_min_mean_neg(disc_output)
        else:
            loss = -torch.mean(disc_output)
        losses.append(loss)

    return torch.mean(torch.stack(losses))
