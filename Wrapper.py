#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''
import torch
import torch.nn as nn
import functools
from Convs import AlignConv, TsmConv
from utiles import as_triple
import re
class AlignWrapper(object):
    """
    #TODO
    # Decorator class for converting convolution modules
    # to corresponding align version in any networks.
    
    # Args:
    #     ref_spacing (int or tuple, optional): Reference spacings ([depth_spacing,] height_spacing, width_spacing). Default: 1
    #     align_inter (str, optional): Interpolation method used in grid_sample. Default: ``bilinear``
    #     memory_efficient (bool, optional): If ``True``, checkpoint the forward function. Default: ``False``
    #     augment (str, optional): Distribution of scales to sample for random scale augmentation. 
    #         Accepted values `identity`, `uniform` and `gaussian`. Default: None
    #     augment_params (tuple, optional): Ignored when `augment` is None or `identity`.
    #         Regarded as (low_bound, high_bound) when `augment` is `uniform`.
    #         Regarded as (mean, standard deviation) when `augment` is `gaussian`.
    #         Default: None
    #     enable_rw (bool, optional): If ``True``, use RWAlignConv. Default: ``False``
    #     confidence_sigma (int or tuple, optional): Sigmas for computing interpolation confidence ([depth,] height, width). Default: 1

    # Examples:
    #     >>> import AlignWrapper
    #     >>> import torchvision
    #     >>> # m is a standard pytorch model
    #     >>> m = torchvision.models.resnet18(True)
    #     >>> m = nn.DataParallel(m)
    #     >>> # after convert, m is using AlignConv
    #     >>> m = AlignWrapper(ref_spacing=(1., 1.), memory_efficient=True)(m)
    #     >>> # now m can be called with an extra argument, `spacing`
    #     >>> input = torch.rand(batch_size, in_channels, height, width)
    #     >>> spacing = torch.rand(batch_size, 2)
    #     >>> out = m(input, spacing)
    """
    def __init__(self, conv_cfg={}, first_conv_channels=1, skip_conv_reg=None):
        # Dict to temporarily store spacing
        # Key -- gpu_id, Value -- spacings scattered to the corresponding gpu
        assert conv_cfg['conv_type'] in ['AlignConv', 'TsmConv']
        self.spacing = {}
        self.first_conv_channels = first_conv_channels
        self.conv_type = conv_cfg.pop('conv_type', None)
        if self.conv_type=='AlignConv':
            self.shift_module = AlignConv
        elif self.conv_type=='TsmConv':
            self.shift_module = TsmConv
        else:
            raise 'conv type unkonwn erro'
        self.conv_cfg = conv_cfg
    def __call__(self, model):
        model = self._convert_model(model)
        if hasattr(model, 'module'):
            model.module.__class__.forward = self._top_forward(model.module.__class__.forward)
        else:
            model.__class__.forward = self._top_forward(model.__class__.forward)
        AlignConv.forward = self._module_forward(AlignConv.forward)
        TsmConv.forward = self._module_forward(TsmConv.forward)

        return model

    def _module_forward(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get spacing from AlignWrapper instance
            return f(z_spacing=self.spacing[torch.cuda.current_device()],
                *args, **kwargs)
        return wrapper

    def _top_forward(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Save spacing in AlignWrapper instance
            if len(args) == 3: # In case spacing is passed as positional argument
                self.spacing[torch.cuda.current_device()] = args[-1]
                args = args[:-1]
            else:              # In case spacing is not passed or passed as keyword argument
                self.spacing[torch.cuda.current_device()] = kwargs.pop('z_spacing', None)

            # Execute the vanilla forward function
            output = f(*args, **kwargs)
            
            # Delete spacing from AlignWrapper instance
            del self.spacing[torch.cuda.current_device()]
            return output
        return wrapper



    def _convert_model(self, module):
        """Traverse the input module and its child recursively
           and replace all instance of torch.nn.modules.conv._Conv*N*d
           to AlignConv*N*d
        """
        if isinstance(module, nn.DataParallel):
            mod = module.module
            mod = self._convert_model(mod)
            mod = nn.DataParallel(mod)
            return mod

        mod = module

        if isinstance(module, nn.modules.conv.Conv2d):
            is_first_conv = module.in_channels==3
            enable_shift_key = self.conv_type.lower().replace('conv', '')
            if is_first_conv:
                in_channels = self.first_conv_channels
                self.conv_cfg.update({enable_shift_key: False})
            else:
                in_channels = module.in_channels
                self.conv_cfg.update({enable_shift_key: True})
            kwargs = dict(in_channels=in_channels, out_channels=module.out_channels,
                            kernel_size=module.kernel_size,
                            stride=module.stride,
                            padding=module.padding,
                            dilation=module.dilation,
                            groups=module.groups,
                            bias=module.bias,
                            padding_mode=module.padding_mode,)
            kwargs.update(self.conv_cfg)
            mod = self.shift_module(**kwargs)
            mod.weight.data = module.weight.unsqueeze(2).data.clone().detach()
            if is_first_conv:
                    a = self.first_conv_channels // 3 + 1
                    b = self.first_conv_channels % 3 + 1 
                    mod.weight.data =  mod.weight.repeat(1,a,1,1,1).data[:,b//2:b//2-b,...]
            if module.bias is not None:
                mod.bias = nn.Parameter(module.bias.data.clone())
                # Decorate forward function for every convolution module
                # mod.forward = self._module_forward(mod.forward)
        elif isinstance(module, (nn.modules.pooling.AvgPool2d, nn.modules.pooling.MaxPool2d)):
            pooling = getattr(nn, module.__class__.__name__.replace('2','3'))
            mod = pooling(kernel_size=as_triple(module.kernel_size, 1),
                            stride=as_triple(module.stride, 1),
                            padding=as_triple(module.padding, 0),)

        elif isinstance(module, (nn.modules.pooling.AdaptiveAvgPool2d, nn.modules.pooling.AdaptiveMaxPool2d)):
            pooling = getattr(nn, module.__class__.__name__.replace('2','3'))
            mod = pooling(output_size=as_triple(module.output_size, 1))

        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            mod = nn.modules.batchnorm.BatchNorm3d(num_features=module.num_features,
                                                        eps=module.eps,
                                                        momentum=module.momentum,
                                                        affine=module.affine,
                                                        track_running_stats=module.track_running_stats,)
            mod.weight.data = module.weight.data.clone().detach()
        else:
            for name, child in module.named_children():
                mod.add_module(name, self._convert_model(child))

        del module
        return mod
