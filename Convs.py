import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
import numpy as np
from utiles import as_triple




class AlignConv(_ConvNd):
    '''
    #TODO
    # Args: 
    #     n_fold (int): Divisor of channels to shift.
    #     shift_method ('tsm', 'align', 'none'): Channnel shift method.
    #     inplace (bool): Enable inplace opertion.
    #     ref_spacing (float): Reference z axis spacing Default: 0.2mm.
    #     align_padding_zero

    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 n_fold=8, align=True, inplace=True,
                 ref_spacing=0.2, shift_padding_zero=True):

        kernel_size = as_triple(kernel_size)
        stride = as_triple(stride)
        padding = as_triple(padding, d_value=0)
        dilation = as_triple(dilation)
        # print(kernel_size, stride,padding, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, as_triple(0,0), groups, bias, padding_mode)
        self.n_fold = n_fold
        self.enable_align = align
        self.inplace = inplace
        self.ref_spacing = ref_spacing
        self.shift_padding_zero = shift_padding_zero

    def align(self, input, fold, z_spacing, padding_zero=False):
        alph = 1 - (self.align_spacing / z_spacing).view(-1, 1 ,1, 1, 1).clamp_(0., 1.)
        out = torch.zeros_like(input)
        out[:, :fold, :-1] = input[:,  :fold, :-1] * alph +  input[:,  :fold, 1:] * (1-alph)
        out[:, fold: 2 * fold, 1:] = input[:,  fold: 2 * fold, 1:] * alph + \
                                             input[:,  fold: 2 * fold, :-1] * (1-alph)
        out[:, 2 * fold:, :] = input[: , 2 * fold:, :]
        if not padding_zero:
            out[:, :fold, -1:] = alph * input[:,  :fold, -1:]
            out[:, fold:2*fold, :1] = alph * input[:, fold:2*fold, :1]        
        return out

    def align_shift(self, x, fold, ref_spacing, z_spacing, padding_zero, inplace):
        if inplace:
            x = inplace_align(x, fold, ref_spacing, z_spacing, padding_zero)
        else:
            x = self.algn(x, fold)
        return x

    def forward(self, input, z_spacing=1.):
        if self.enable_align:
            _, c, _, _, _ = input.size()
            fold = c // self.n_fold 
            input = self.align_shift(input, fold, self.ref_spacing, \
                z_spacing, self.shift_padding_zero, self.inplace)
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups) 

class TsmConv(_ConvNd):
    '''
    #TODO
    # Args: 
    #     n_fold (int): Divisor of channels to shift.
    #     shift_method ('tsm', 'align', 'none'): Channnel shift method.
    #     inplace (bool): Enable inplace opertion.
    #     ref_spacing (float): Reference z axis spacing Default: 0.2mm.
    #     align_padding_zero

    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 n_fold=8, tsm=True, inplace=True,
                 shift_padding_zero=True):

        kernel_size = as_triple(kernel_size)
        stride = as_triple(stride)
        padding = as_triple(padding, d_value=0)
        dilation = as_triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, as_triple(0,0), groups, bias, padding_mode)
        self.n_fold = n_fold
        self.enable_tsm = tsm
        self.inplace = inplace
        self.shift_padding_zero = shift_padding_zero

    def tsm(self, input, fold):
        out = torch.zeros_like(input)
        out[:, :fold, :-1] = input[:,  :fold, 1:]
        out[:, fold: 2 * fold, 1:] = input[:,  fold: 2 * fold, :-1]
        out[:, 2 * fold:, :] = input[: , 2 * fold:, :]       
        return out

    def tsm_shift(self, x, fold, inplace):
        if inplace:
            x = inplace_tsm(x, fold)
        else:
            x = self.tsm(x, fold)
        return x

    def forward(self, input):
        if self.enable_tsm:
            _, c, _, _, _ = input.size()
            fold = c // self.n_div     
            input = self.tsm_shift(input, fold, self.inplace)
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups) 

class InplaceTsm(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        '''
        @params: 
            input: BxCxDxHxW
          n_div: xx
        '''
        # not support higher order gradient
        # input.detach_()
        n, c, t, h, w = input.size()
        # fold = c//n_div
        ctx.fold_ = fold
        buffer = input.data.new(n, fold, t, h, w).zero_()
        buffer[:, :, :-1] = input.data[:, :fold, 1:]
        input.data[:, :fold, :] = buffer
        buffer.zero_()
        buffer[:, :, 1:] = input.data[:, fold: 2 * fold, :-1]
        input.data[:, fold: 2 * fold, :] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output.detach_()
        fold = ctx.fold_
        n, c, t, h, w = grad_output.size()
        buffer = grad_output.data.new(n, fold, t, h, w).zero_()
        buffer[:, :, 1:] = grad_output.data[:, :fold, :-1]
        grad_output.data[:, :fold, :] = buffer
        buffer.zero_()
        buffer[:, :, :-1] = grad_output.data[:, fold: 2 * fold, 1:]
        grad_output.data[:, fold: 2 * fold, :] = buffer
        return grad_output, None


class InplaceAlign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fold, align_spacing, z_spacing, padding_zero=True):
        '''
        @params: 
            input: BxCxDxHxW
            fold: channels to align
            align_spacing: align spacing
            z_spacing: z_spacing of the input ct
            padding_zero: bool, whether padding zero to outer cts 
        '''
        n, c, t, h, w = input.size()
        ctx.fold_ = fold
        ctx.padding_zero = padding_zero
        alph = 1 - (align_spacing / z_spacing).view(-1, 1 ,1, 1, 1).clamp_(0., 1.) ##把小于align_spacing的当作align_spacing，不做插值处理 
        ctx.alph_ = alph
        input.data[:, :fold, :-1] = input.data[:, :fold, :-1] * alph + input.data[:, :fold, 1:] * (1-alph)
        input.data[:, fold:2*fold, 1:] = input.data[:, fold:2*fold, 1:] * alph + \
                                                                input.data[:, fold:2*fold, :-1] * (1-alph)
        if padding_zero:
            input.data[:, :fold, -1:] =  alph * input.data[:, :fold, -1:]
            input.data[:, fold:2*fold, :1] = alph * input.data[:, fold:2*fold, :1]
        return input

    @staticmethod
    def backward(ctx, grad_output):
        fold = ctx.fold_
        alph = ctx.alph_
        padding_zero = ctx.padding_zero
        n, c, t, h, w = grad_output.size()
        pad_fold = alph if padding_zero else 1.      
        grad_output.data[:, :fold, -1:] = grad_output.data[:, :fold, -1:] \
                                                * pad_fold + grad_output.data[:, :fold, -2:-1] * (1-alph)
        grad_output.data[:, :fold, 1:-1] = grad_output.data[:, :fold, 1:-1] * alph\
                                                             + grad_output.data[:, :fold, :-2] * (1-alph)
        grad_output.data[:, :fold, :1] = grad_output.data[:, :fold, :1] * alph       
        grad_output.data[:, fold:2*fold, :1] = grad_output.data[:, fold:2*fold, :1] * pad_fold\
                                                        + grad_output.data[:, fold:2*fold, 1:2] * (1-alph)                                            
        grad_output.data[:, fold:2*fold, 1:-1] = grad_output.data[:, fold:2*fold, 1:-1] * alph\
                                                        + grad_output.data[:, fold:2*fold, 2:] * (1-alph)                                    
        grad_output.data[:, fold:2*fold, -1:] = grad_output.data[:, fold:2*fold, -1:] * alph
        return grad_output, None, None, None, None

inplace_tsm = InplaceTsm.apply
inplace_align = InplaceAlign.apply
