import torch
import torch.nn as nn

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        ifft_shape_slice = x.shape[-2:]

        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.bn(ffted)
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous() 
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=(-2,-1), norm='ortho')
        return output

class SpectralTransform(nn.Module):
    def __init__(self, channels, **fu_kwargs):
        super(SpectralTransform, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(channels // 2, channels // 2, **fu_kwargs)

        self.conv2 = nn.Conv2d(channels // 2, channels, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output

class FFC(nn.Module):

    def __init__(self, channels, ratio):
        super(FFC, self).__init__()

        cg = int(channels * ratio)
        cl = channels - cg

        self.convl2l = nn.Conv2d(cl, cl, 3, 1, 1, bias=False, padding_mode='reflect')
        self.convl2g = nn.Conv2d(cl, cg, 3, 1, 1, bias=False, padding_mode='reflect')
        self.convg2l = nn.Conv2d(cg, cl, 3, 1, 1, bias=False, padding_mode='reflect')
        self.convg2g = SpectralTransform(cg, cg, 1, 1, False)

    def forward(self, x):
        x_l, x_g = x
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg

class FFC_BN_ACT_local(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=False):
        super(FFC_BN_ACT_local, self).__init__()

        self.conv_l = nn.Conv2d(in_channels, out_channels, 
            kernel_size, stride, padding,  
            bias=bias, padding_mode='reflect')
        self.bn_l = nn.BatchNorm2d(out_channels)
        self.act_l = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l = self.conv_l(x)
        x_l = self.bn_l(x_l)
        x_l = self.act_l(x_l)
        return x_l

class FFC_BN_ACT_local2global(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=False, ratio_gout=0.75):
        super(FFC_BN_ACT_local2global, self).__init__()

        out_cl = int(out_channels * (1 - ratio_gout))
        out_cg = int(out_channels * ratio_gout)

        self.conv_l2l = nn.Conv2d(in_channels, out_cl, 
            kernel_size, stride, padding,  
            bias=bias, padding_mode='reflect')
        self.bn_l2l = nn.BatchNorm2d(out_cl)
        self.act_l2l = nn.ReLU(inplace=True)

        self.conv_l2g = nn.Conv2d(in_channels, out_cg, 
            kernel_size, stride, padding,  
            bias=bias, padding_mode='reflect')
        self.bn_l2g = nn.BatchNorm2d(out_cg)
        self.act_l2g = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l = self.conv_l2l(x)
        x_l = self.bn_l2l(x_l)
        x_l = self.act_l2l(x_l)

        x_g = self.conv_l2g(x)
        x_g = self.bn_l2g(x_g)
        x_g = self.act_l2g(x_g)
        return x_l, x_g

class FFC_BN_ACT_global(nn.Module):

    def __init__(self, channels, ratio):
        super(FFC_BN_ACT_global, self).__init__()
        self.ffc = FFC(channels, ratio)

        global_channels = int(channels * ratio)
        self.bn_l = nn.BatchNorm2d(channels - global_channels)
        self.bn_g = nn.BatchNorm2d(global_channels)
        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)

class FFCResnetBlock(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        self.conv1 = FFC_BN_ACT_global(channels, ratio)
        self.conv2 = FFC_BN_ACT_global(channels, ratio)

    def forward(self, x):
        x_l, x_g = x
        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        return x_l, x_g

class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ratio=0.75, ngf=64, n_sampling=3, n_blocks=9):
        super().__init__()

        model = [nn.ReflectionPad2d(3), FFC_BN_ACT_local(input_nc, ngf, 7, 1, 0)]

        ### downsample
        for i in range(n_sampling):
            mult = 2 ** i
            if i == n_sampling - 1:
                model += [FFC_BN_ACT_local2global(ngf*mult, ngf*mult*2, 3, 2, 1, ratio_gout=ratio)]
            else:
                model += [FFC_BN_ACT_local(ngf*mult, ngf*mult*2, 3, 2, 1)]

        ### resnet blocks
        mult = 2 ** n_sampling
        for i in range(n_blocks):
            model += [FFCResnetBlock(ngf*mult, ratio)]
        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_sampling):
            mult = 2 ** (n_sampling - i)
            model += [
                nn.ConvTranspose2d(ngf*mult,  int(ngf*mult // 2), 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(int(ngf*mult // 2)),
                nn.ReLU(inplace=True)
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, 1, 0)]
        model.append(nn.Sigmoid())

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)