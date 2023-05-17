""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from .helpers import to_2tuple
from .trace_utils import _assert
import torch_dwt as tdwt

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
            dwt_level=[0, 0, 0],
            dwt_kernel_size=[0, 0, 0],        
    ):
        super().__init__()
        
        self.dwt_kernel_size = dwt_kernel_size
        self.dwt_level = dwt_level
        dwt_conv_layer = list()
        #print('@@@@@@@@ DWT Level {}'.format(self.dwt_level))
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if self.dwt_level[0] == 0:
                #self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        else:
            if self.dwt_level[0] == 1:
                #print('DWT _LEVEL 1 OK ')
                for idx in range(4):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias).cuda())
            elif self.dwt_level[0] == 2:
                #print('DWT _LEVEL 2 OK ')
                for idx in range(16):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, embed_dim, kernel_size=8, stride=8, bias=bias).cuda())
            else:
                for idx in range(16):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size/16, stride=patch_size/16, bias=bias).cuda())
            self.dwt_conv_layer = nn.ModuleList(dwt_conv_layer)
                
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

                                              
    #Renew    
    def dwt_rearrange(self, if_map, dwt_ratio, post_norm=False, dwt_quant=1, dwt_drop=False):
        split_tensor_lst = list()
                                              
        if self.dwt_level[0] == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst, x_dwt_rate=dwt_ratio)
        elif self.dwt_level[0] == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst, x_dwt_rate=dwt_ratio, x_quant=dwt_quant)
            #print('split_tensor_lst shape', split_tensor_lst[0].shape)
        else:
            tdwt.get_dwt_level2(if_map, split_tensor_lst, x_dwt_rate=dwt_ratio)
        
        module_lst1 = [(self.dwt_conv_layer[i]) for i in range(len(split_tensor_lst))]
        input_lst1 = [((split_tensor_lst[i])) for i in range(len(split_tensor_lst))]
        output_tensor_lst = nn.parallel.parallel_apply(module_lst1, input_lst1)
        
        if self.dwt_level[0] == 1:
            return tdwt.get_dwt_level1_inverse(output_tensor_lst)
        elif self.dwt_level[0] == 2:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst, 2)
        else:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst)
                                              
                                              
    def forward(self, x, dwt_quant=1):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        #print('Before projection data shape --> ', x.shape)
        
        if self.dwt_level[0] == 0:
            x = self.proj(x)
            #print('check forward -> ', x.shape)
        else:
            x = self.dwt_rearrange(x, None, dwt_quant=1)
            #print('check forward -> ', x.shape)
                                              
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
                                                  
    def init_weights(self):
        if self.dwt_kernel_size[0] != 0:
            for idx, item in enumerate(self.dwt_conv_layer):
                nn.init.kaiming_normal_(item.weight, mode='fan_out', nonlinearity='relu')