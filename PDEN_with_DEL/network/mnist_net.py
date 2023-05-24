import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dwt as tdwt
class ConvNet(nn.Module):
    ''' 网络结构和cvpr2020的 M-ADA 方法一致 '''
    def __init__(self, imdim=3):
        super(ConvNet, self).__init__()
        inplace_available= False
        
        
        self.dwt_level = 2
        self.dwt_kernel_size = 2
        self.quant = 1
        
        
        inplanes = 64
        dwt_conv_layer = list()
        if self.dwt_level == 1:
            self.fc1 = nn.Linear(2304, 1024)
            for idx in range(4):
                dwt_conv_layer.append(nn.Conv2d(imdim, inplanes, kernel_size=self.dwt_kernel_size, stride=1, padding=0, bias=False).cuda())
        elif self.dwt_level == 2:
            self.drop_layer = nn.Dropout2d(p=0.5)
            if self.dwt_kernel_size == 1:
                self.fc1 = nn.Linear(4608, 1024)
            else:
                self.fc1 = nn.Linear(3200, 1024)
            for idx in range(16):
                dwt_conv_layer.append(nn.Conv2d(imdim, inplanes, kernel_size=self.dwt_kernel_size, stride=1, padding=0, bias=False).cuda())
        elif self.dwt_level == 3:
            self.fc1 = nn.Linear(256, 1024)
            for idx in range(64):
                dwt_conv_layer.append(nn.Conv2d(imdim, inplanes, kernel_size=self.dwt_kernel_size, stride=1, padding=0, bias=False).cuda())
        else:
            self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
            self.fc1 = nn.Linear(3200, 1024)
        self.dwt_conv_layer = nn.ModuleList(dwt_conv_layer)
        
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)        
        self.relu2 = nn.ReLU()
        
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU()
        
        self.cls_head_src = nn.Linear(1024, 10)
        self.cls_head_tgt = nn.Linear(1024, 10)
        self.pro_head = nn.Linear(1024, 128)
            
    def conv(self, x, c_layer, isLow=True):
        c_layer_weight = c_layer.weight.clone()
        x = nn.functional.conv2d(x, c_layer_weight, bias=c_layer.bias)
        return x

        if isLow == False:
            layer_norm = nn.LayerNorm([x.shape[-2],x.shape[-1]]).cuda()
            return layer_norm(x)
        else:
            layer_norm = nn.LayerNorm([x.shape[-2],x.shape[-1]]).cuda()
            return layer_norm(x)
       
        
    def dwt_rearrange(self, if_map, dwt_ratio=None, post_norm=False):
        split_tensor_lst = list()
        
        if self.dwt_level == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst, x_dwt_rate=self.quant)
        elif self.dwt_level == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst, x_dwt_rate=self.quant)
        elif self.dwt_level == 3:
            tdwt.get_dwt_level3(if_map, split_tensor_lst, x_dwt_rate=self.quant)
        else:
            tdwt.get_dwt_level2(if_map, split_tensor_lst, x_dwt_rate=self.quant)
      
        module_lst1 = [(self.dwt_conv_layer[i]) for i in range(len(split_tensor_lst))]
        
        input_lst1 = [((split_tensor_lst[i])) for i in range(len(split_tensor_lst))]
        
        #output_tensor_lst = nn.parallel.parallel_apply(module_lst1, input_lst1)
        output_tensor_lst = list()
        for i in range(len(split_tensor_lst)):
            if i==0 or i==4 or i == 8 or i==12: #Low passband 
                #output_tensor_lst.append(self.conv(self.drop_layer(split_tensor_lst[i]), self.dwt_conv_layer[i]))
                output_tensor_lst.append(self.conv(split_tensor_lst[i], self.dwt_conv_layer[i],isLow=True))
            else: #high passband
                #output_tensor_lst.append(self.conv(self.drop_layer(split_tensor_lst[i]), self.dwt_conv_layer[i]))
                output_tensor_lst.append(self.conv(split_tensor_lst[i], self.dwt_conv_layer[i],isLow=False))
        if self.dwt_level == 1:
            return tdwt.get_dwt_level1_inverse(output_tensor_lst)
        elif self.dwt_level == 2:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst)
        elif self.dwt_level == 3:
            return tdwt.get_dwt_level3_inverse(output_tensor_lst)
        else:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst)        
        
    def forward(self, x, mode='test'):

        in_size = x.size(0)
        #print('IN size is ',in_size)
        if self.dwt_level == 0:
            conv1_weight = self.conv1.weight.clone()
            x = nn.functional.conv2d(x, conv1_weight, bias=self.conv1.bias)
        else:
            x = self.dwt_rearrange(x)
            x = self.drop_layer(x)

        x = self.mp(self.relu1(x))
        
        conv2_weight = self.conv2.weight.clone()
        x = nn.functional.conv2d(x, conv2_weight, bias=self.conv2.bias)
        
            
        x = self.mp(self.relu2(x))
        x = x.view(in_size, -1)
        
        fc1_weight = self.fc1.weight.clone()
        x = nn.functional.linear(x, fc1_weight, bias=self.fc1.bias)
        x = self.relu3(x)
        fc2_weight = self.fc2.weight.clone()
        x = nn.functional.linear(x, fc2_weight, bias=self.fc2.bias)
        x = self.relu4(x)
        
        if mode == 'test':
            p = self.cls_head_src(x)
            return p
        elif mode == 'train':
            weight_p = self.cls_head_src.weight.clone()
            p = nn.functional.linear(x, weight_p, bias=self.cls_head_src.bias)

            weight = self.pro_head.weight.clone()
            z = nn.functional.linear(x, weight, bias=self.pro_head.bias)
            z = F.normalize(z)
            return p,z
        elif mode == 'p_f':
            p = self.cls_head_src(out4)
            return p, out4
            
class ConvNetVis(nn.Module):
    ''' 方便可视化，特征提取器输出2-d特征
    '''
    def __init__(self, imdim=3):
        super(ConvNetVis, self).__init__()
        inplace_available= False
        self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=inplace_available)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=inplace_available)
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.relu3 = nn.ReLU(inplace=inplace_available)
        self.fc2 = nn.Linear(1024, 2)
        self.relu4 = nn.ReLU(inplace=inplace_available)
        
        self.cls_head_src = nn.Linear(2, 10)
        self.cls_head_tgt = nn.Linear(2, 10)
        self.pro_head = nn.Linear(2, 128)

    def forward(self, x, mode='test'):

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))
        
        if mode == 'test':
            p = self.cls_head_src(out4)
            return p
        elif mode == 'train':
            p = self.cls_head_src(out4)
            z = self.pro_head(out4)
            z = F.normalize(z)
            return p,z
        elif mode == 'p_f':
            p = self.cls_head_src(out4)
            return p, out4
        #elif mode == 'target':
        #    p = self.cls_head_tgt(out4)
        #    z = self.pro_head(out4)
        #    z = F.normalize(z)
        #    return p,z
    

