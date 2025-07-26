import torch
import torch.nn as nn
from SelfONN import SuperONN1d, SuperONN2d, SelfONN1d, SelfONN2d


class SelfResNet(nn.Module):
    
    def __init__(self, input_ch, class_num, q_order, in_ft=8, BatchNorm=False): 
        super(SelfResNet, self).__init__()  
        self.BatchNorm = BatchNorm
        # Initial Conv layer     out_dim = [(in_dim+2p-k)/s]+1 
        self.conv1  = SelfResNet.conv_block(in_channels=input_ch, out_channels=in_ft, kernel_size=7, stride=2, padding=3, q_order=q_order, bias=False, BatchNorm=self.BatchNorm)
        self.pool1  = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # 1st Residual Block     out_dim = in_dim
        self.res_block_1_1 = Self_residual_block(in_channels=in_ft, out_channels=in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm)
        self.res_block_1_2 = Self_residual_block(in_channels=in_ft, out_channels=in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm)
        # 2nd Residual Block     out_dim = in_dim/2
        self.res_block_2_1 = Self_residual_block(in_channels=in_ft, out_channels=2*in_ft, kernel_size=3, stride_1=2, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=True, BatchNorm=self.BatchNorm)
        self.res_block_2_2 = Self_residual_block(in_channels=2*in_ft, out_channels=2*in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm)
        # 3rd Residual Block     out_dim = in_dim/2
        self.res_block_3_1 = Self_residual_block(in_channels=2*in_ft, out_channels=4*in_ft, kernel_size=3, stride_1=2, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=True, BatchNorm=self.BatchNorm)
        self.res_block_3_2 = Self_residual_block(in_channels=4*in_ft, out_channels=4*in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm)
        # 4th Residual Block     out_dim = in_dim/2
        self.res_block_4_1 = Self_residual_block(in_channels=4*in_ft, out_channels=8*in_ft, kernel_size=3, stride_1=2, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=True, BatchNorm=self.BatchNorm)
        self.res_block_4_2 = Self_residual_block(in_channels=8*in_ft, out_channels=8*in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm)
        # Average pooling        out_dim = 8*in_ft
        self.AvgPool = torch.nn.AdaptiveAvgPool1d(output_size=1) 
        self.flatten = torch.nn.Flatten() 
        # Output MLP layer 
        self.MLP1 = torch.nn.Linear(in_features=8*in_ft, out_features=class_num, bias=True)
        self.softmax = torch.nn.LogSoftmax(dim=1) 

    def forward(self, x):
        conv_1 = self.conv1(x)
        pool1_1 = self.pool1(conv_1)
        res_1 = self.res_block_1_2(self.res_block_1_1(pool1_1))
        res_2 = self.res_block_2_2(self.res_block_2_1(res_1))
        res_3 = self.res_block_3_2(self.res_block_3_1(res_2))
        res_4 = self.res_block_4_2(self.res_block_4_1(res_3))
        AdaptivePool = self.flatten(self.AvgPool(res_4))
        Output_layer = self.MLP1(AdaptivePool)  
        return self.softmax(Output_layer) 

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, q_order=1, bias=False, BatchNorm=False):      
        if BatchNorm:
            return nn.Sequential(
                SelfONN1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),             
                nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Tanh() 
            )
        else: 
            return nn.Sequential(
                SelfONN1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),             
                nn.Tanh() 
            )


class SelfResAttentioNet(nn.Module):
    
    def __init__(self, input_ch, class_num, q_order, in_ft=8, BatchNorm=False): 
        super(SelfResAttentioNet, self).__init__()  
        self.BatchNorm = BatchNorm
        # Initial Conv layer     out_dim = [(in_dim+2p-k)/s]+1 
        self.conv1  = SelfResNet.conv_block(in_channels=input_ch, out_channels=in_ft, kernel_size=7, stride=2, padding=3, q_order=q_order, bias=False, BatchNorm=self.BatchNorm)
        self.pool1  = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # 1st Residual Block     out_dim = in_dim
        self.res_block_1_1 = Self_residual_block(in_channels=in_ft, out_channels=in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm)
        self.res_block_1_2 = Self_residual_block(in_channels=in_ft, out_channels=in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm, Atttention=True)
        # 2nd Residual Block     out_dim = in_dim/2
        self.res_block_2_1 = Self_residual_block(in_channels=in_ft, out_channels=2*in_ft, kernel_size=3, stride_1=2, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=True, BatchNorm=self.BatchNorm)
        self.res_block_2_2 = Self_residual_block(in_channels=2*in_ft, out_channels=2*in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm, Atttention=True)
        # 3rd Residual Block     out_dim = in_dim/2
        self.res_block_3_1 = Self_residual_block(in_channels=2*in_ft, out_channels=4*in_ft, kernel_size=3, stride_1=2, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=True, BatchNorm=self.BatchNorm)
        self.res_block_3_2 = Self_residual_block(in_channels=4*in_ft, out_channels=4*in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm, Atttention=True)
        # 4th Residual Block     out_dim = in_dim/2
        self.res_block_4_1 = Self_residual_block(in_channels=4*in_ft, out_channels=8*in_ft, kernel_size=3, stride_1=2, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=True, BatchNorm=self.BatchNorm)
        self.res_block_4_2 = Self_residual_block(in_channels=8*in_ft, out_channels=8*in_ft, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=q_order, bias=False, downsample=False, BatchNorm=self.BatchNorm, Atttention=True)
        # Average pooling        out_dim = 8*in_ft
        self.AvgPool = torch.nn.AdaptiveAvgPool1d(output_size=1) 
        self.flatten = torch.nn.Flatten() 
        # Output MLP layer 
        self.MLP1 = torch.nn.Linear(in_features=8*in_ft, out_features=class_num, bias=True)
        self.softmax = torch.nn.LogSoftmax(dim=1) 

    def forward(self, x):
        conv_1 = self.conv1(x)
        pool1_1 = self.pool1(conv_1)
        res_1 = self.res_block_1_2(self.res_block_1_1(pool1_1))
        res_2 = self.res_block_2_2(self.res_block_2_1(res_1))
        res_3 = self.res_block_3_2(self.res_block_3_1(res_2))
        res_4 = self.res_block_4_2(self.res_block_4_1(res_3))
        AdaptivePool = self.flatten(self.AvgPool(res_4))
        Output_layer = self.MLP1(AdaptivePool)  
        return self.softmax(Output_layer) 

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, q_order=1, bias=False, BatchNorm=False):      
        if BatchNorm:
            return nn.Sequential(
                SelfONN1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),             
                nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Tanh() 
            )
        else: 
            return nn.Sequential(
                SelfONN1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),             
                nn.Tanh() 
            )


class Self_residual_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride_1=1, stride_2=1, padding=1, q_order=1, bias=False, downsample=False, BatchNorm=False, Atttention = False): 
        super(Self_residual_block, self).__init__()
        self.conv1 = SelfONN1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride_1,padding=padding,dilation=1,groups=1,bias=True,q=q_order,mode='fast')            
        self.bn1 = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.Tanh = nn.Tanh() 
        self.conv2 = SelfONN1d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride_2,padding=padding,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.bn2 =nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.BatchNorm = BatchNorm
        if downsample: 
            if self.BatchNorm:
                self.downsample = nn.Sequential( 
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=bias), 
                nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ) 
            else:
                self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=bias)
        else:
            self.downsample = downsample
        self.Atttention = Atttention

    def attention_(self, input):
        embed_shape = input.shape[-1]
        attention_layer = nn.MultiheadAttention(embed_dim=embed_shape, num_heads=2, batch_first=True)
        return attention_layer    
        

    def forward(self, x):
        identity = x
        if self.BatchNorm:
            # 1st conv layer
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.Tanh(out)
            # 2nd conv layer
            out = self.conv2(out)
            out = self.bn2(out)
        else:
            # 1st conv layer
            out = self.conv1(x)
            out = self.Tanh(out)
            # 2nd conv layer
            out = self.conv2(out)
        # downsample residual path feature map
        if self.downsample: 
            identity = self.downsample(x)
        # residual path
        if self.Atttention:
            self.attention = self.attention_(identity).cuda()
            atten,_ = self.attention(identity, out, identity)
            out += atten
        else:
            out += identity
        out = self.Tanh(out)
        return out
