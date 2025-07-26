
import torch
import torch.nn as nn
from SelfONN import SuperONN1d, SelfONN1d


class UnSqueezeLayer(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        return x


class SqueezeLayer(nn.Module):
    def forward(self, x):
        x = x.squeeze(-1)
        # x = x.squeeze(2)
        return x
    
    
class CNN_Classifier(nn.Module):
    def __init__(self, in_channels, class_num, final_activation='LogSoftmax'):
        super().__init__()
        if final_activation == 'LogSigmoid':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.LogSigmoid()
                )
        elif final_activation == 'LogSoftmax':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num), 
                nn.LogSoftmax(dim=1)
                )
        elif final_activation == 'Sigmoid':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.Sigmoid()
                )
        elif final_activation == 'Softmax':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.Softmax(dim=1)
                )
        elif final_activation == 'Softsign':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.Softsign()
                )
        elif final_activation == 'Tanh':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.Tanh()
                )
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.fill_(0.01) 
     
    def forward(self,x):
        x = self.classifier(x)
        return x


class Self_B_ResBlock(nn.Module):
    def __init__(self, in_channels=3, channel1=8, channel2=16, channel3=8, resConnection=False, concat =False, dropout=None):
        super().__init__()
        self.layer1 = SelfONN1d(in_channels=in_channels,out_channels=channel1,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,q=3,mode='fast')
        self.Batch1 = nn.BatchNorm1d(channel1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.concat = concat
        self.resConnection = resConnection
        self.dropout = dropout
        if self.resConnection:
            self.layer2 = SelfONN1d(in_channels=channel1,out_channels=channel2,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=3,mode='fast')
            self.Batch2 = nn.BatchNorm1d(channel2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        else:
            self.layer2 = SelfONN1d(in_channels=channel1,out_channels=channel2,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=3,mode='fast')
            self.Batch2 = nn.BatchNorm1d(channel2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = SelfONN1d(in_channels=channel2,out_channels=channel3,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,q=3,mode='fast')
        self.Batch3 = nn.BatchNorm1d(channel3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()

    def forward(self,x):
        input = x.clone()
        x = self.tanh(self.Batch1(self.layer1(x)))
        x = self.tanh(self.Batch2(self.layer2(x)))
        x = self.tanh(self.Batch3(self.layer3(x)))
        if self.resConnection:
          if self.concat:
              x = self.tanh(torch.cat((x,input), 1))
          else:
              x= self.tanh(x.clone()+ input)
        return x


class Self_MobileNet(nn.Module):
    def __init__(self, input_channel = 3, last_layer_channel = 32, class_num= 10):
        super().__init__()
        self.class_num = class_num
        self.selfONN = SelfONN1d(in_channels=input_channel,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=3,mode='fast')
        self.batchnorm = nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.BottleneckResidual1 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True, concat=False)
        self.avgpool1d = torch.nn.AvgPool1d(kernel_size=2, stride=1, padding=0)
        self.BottleneckResidual2 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True, concat=False)
        self.BottleneckResidual3 = Self_B_ResBlock(in_channels=32, channel1=48, channel2=48, channel3=32, resConnection=True, concat=False)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool1d(12)
        self.Flatten = nn.Flatten()
        self.Dropout1d = nn.Dropout(p=0.2)
        self.self_MLP = CNN_Classifier(in_channels = int(12*last_layer_channel), class_num = self.class_num)
        
    def forward(self, x):
        x= self.tanh(self.batchnorm(self.selfONN(x)))
        x= self.BottleneckResidual1(x)
        x = self.avgpool1d(x)
        x= self.BottleneckResidual2(x)
        x = self.avgpool1d(x)
        x= self.BottleneckResidual3(x)
        x= self.AdaptiveAvgPool2d(x)
        x = self.Flatten(x)
        x = self.Dropout1d(x)
        x= self.self_MLP(x)
        return x


class Self_DenseMobileNet(nn.Module):
    def __init__(self, input_channel = 3, last_layer_channel = 32, class_num= 10):
        super().__init__()
        self.class_num = class_num
        self.InputMLP = 10
        self.selfONN = SelfONN1d(in_channels=input_channel,out_channels=32,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=3,mode='fast')
        self.batchnorm = nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.BottleneckResidual1 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True)
        self.BottleneckResidual2 = Self_B_ResBlock(in_channels=32, channel1=72, channel2=72, channel3=32, resConnection=True)
        self.maxpool = nn.MaxPool1d(2)
        self.BottleneckResidual3 = Self_B_ResBlock(in_channels=32, channel1=84, channel2=84, channel3=32, resConnection=True)
        self.BottleneckResidual4 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True)
        self.BottleneckResidual5 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True)
        self.BottleneckResidual6 = Self_B_ResBlock(in_channels=32, channel1=84, channel2=84, channel3=32, resConnection=True)
        self.BottleneckResidual7 = Self_B_ResBlock(in_channels=32, channel1=72, channel2=72, channel3=32, resConnection=True)
        self.BottleneckResidual8 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True)
        self.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        self.Flatten = nn.Flatten()
        self.Dropout = nn.Dropout(p=0.2)
        self.self_MLP = CNN_Classifier(in_channels =6*32, class_num = self.class_num)
    
    def forward(self,x):
        x = self.tanh(self.batchnorm(self.selfONN(x)))
        x = self.BottleneckResidual1(x)
        x = self.BottleneckResidual2(x)
        ##############
        inMLP1 = self.Flatten(self.AdaptiveAvgPool1d(x)) # input for MLP

        x = self.BottleneckResidual3(self.maxpool(x))
        x = self.BottleneckResidual4(x)
        ##############
        inMLP2 = self.Flatten(self.AdaptiveAvgPool1d(x)) # input for MLP

        x = self.BottleneckResidual5(self.maxpool(x))
        x = self.BottleneckResidual6(x)
        ##############
        inMLP3 = self.Flatten(self.AdaptiveAvgPool1d(x)) # input for MLP

        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        ##############
        inMLP4 = self.Flatten(self.AdaptiveAvgPool1d(x)) # input for MLP

        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        ##############
        inMLP5 = self.Flatten(self.AdaptiveAvgPool1d(x)) # input for MLP

        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        ##############
        inMLP6 = self.Flatten(self.AdaptiveAvgPool1d(x)) # input for MLP
        inMLP = torch.cat((inMLP1, inMLP2, inMLP3, inMLP4, inMLP5, inMLP6), dim=1)
        x= self.Flatten(inMLP)
        x = self.Dropout(x)
        x= self.self_MLP(x)
        return x


class StackCNN(nn.Module):
    def __init__(self,in_channels=1, out_channels=1, features=8, q=1, learnable=True,max_shift=2,rounded_shifts=False):
        super(StackCNN, self).__init__()
    
        self.Net_1 = Super_ResNet(in_channels=in_channels, features=features, q=q, learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts)
        self.Net_2 = Super_ResNet(in_channels=in_channels, features=features, q=q, learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts)
        self.Net_3 = Super_ResNet(in_channels=in_channels, features=features, q=q, learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts)
        self.fc = torch.nn.Linear(in_features=3*88,out_features=out_channels)

    def forward(self, x):
        x1 = self.Net_1(x[:,0,:].unsqueeze(1))
        x2 = self.Net_2(x[:,1,:].unsqueeze(1))
        x3 = self.Net_3(x[:,2,:].unsqueeze(1))
        x = torch.cat((x1,x2,x3), dim=1)
        return self.fc(x)


class Self_Net(nn.Module):
    def __init__(self, in_channels, features, q=1):
        super(Self_Net, self).__init__()

        self.Net = nn.Sequential(
            # 1st layer (1D CNN)  
            SelfONN1d(in_channels=in_channels,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,bias=True,q=q,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SelfONN1d(in_channels=16,out_channels=features,kernel_size=3,stride=1,padding=1,dilation=1,bias=True,q=q,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(features),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten()
        )  
        
    def forward(self, x):
        return self.Net(x)


class Super_Net(nn.Module):
    def __init__(self, in_channels, features, q=1, learnable=True, max_shift=2,rounded_shifts=False):
        super(Super_Net, self).__init__()

        self.Net = nn.Sequential(
            # 1st layer (1D CNN)  
            SuperONN1d(in_channels=in_channels,out_channels=16,kernel_size=3,q=q,bias=True,padding=1,dilation=1,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SuperONN1d(in_channels=16,out_channels=features,kernel_size=3,q=q,bias=True,padding=1,dilation=1,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(features),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten()
        )  

    def forward(self, x):
        return self.Net(x)


class Self_ResNet(nn.Module):
    def __init__(self, in_channels, features, q=1):
        super(Self_ResNet, self).__init__()

        self.Net = nn.Sequential(
            # 1st layer (1D CNN)  
            SelfResBlock(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, q=q,bias=True,padding=1,dilation=1,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SelfResBlock(in_channels=16, out_channels=features, kernel_size=3, stride=1, q=q,bias=True,padding=1,dilation=1,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(features),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten()
        )  
    def forward(self, x):
        return self.Net(x)


class Super_ResNet(nn.Module):
    def __init__(self, in_channels, features, q=1, learnable=True,max_shift=2,rounded_shifts=False):
        super(Super_ResNet, self).__init__()

        self.Net = nn.Sequential(
            # 1st layer (1D CNN)  
            SuperResBlock(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, q=q,bias=True,padding=1,dilation=1,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SuperResBlock(in_channels=16, out_channels=features, kernel_size=3, stride=1, q=q,bias=True,padding=1,dilation=1,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(features),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten()
        )  
        
    def forward(self, x):
        return self.Net(x)


class Self_MultiResNet(nn.Module):
    def __init__(self, in_channels, q=1):
        super(Self_MultiResNet, self).__init__()

        self.Net = nn.Sequential(
            # 1st layer (1D CNN)  
            # out_channels = 3*in_features = 3*4 = 12
            Self_MultiResBlock(in_channels=in_channels, in_features=4, kernel_size=3, stride=1, q=q,bias=True,padding=1,dilation=1,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(12), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            # out_channels = 3*in_features =3*4 = 12
            Self_MultiResBlock(in_channels=12, in_features=4, kernel_size=3, stride=1, q=q,bias=True,padding=1,dilation=1,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(12),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten()
        )  
    def forward(self, x):
        return self.Net(x)


class Super_MultiResNet(nn.Module):
    def __init__(self, in_channels, q=1, learnable=True,max_shift=2,rounded_shifts=False):
        super(Super_MultiResNet, self).__init__()

        self.Net = nn.Sequential(
            # 1st layer (1D CNN)  
            # out_channels = 3*in_features = 3*4 = 12
            Super_MultiResBlock(in_channels=in_channels, in_features=4, kernel_size=3, stride=1, q=q,bias=True,padding=1,dilation=1,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(12), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            # out_channels = 3*in_features =3*4 = 12
            Super_MultiResBlock(in_channels=12, in_features=4, kernel_size=3, stride=1, q=q,bias=True,padding=1,dilation=1,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(12),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten()
        )  
    def forward(self, x):
        return self.Net(x)

################################## Blocks ##################################

class SelfResBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, q=1,bias=True,padding=1,dilation=1,dropout=None):
        super(SelfResBlock, self).__init__()

        self.Self_1 = SelfONN1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=1,bias=True,q=q,dropout=dropout)
        self.Batch_1 = torch.nn.BatchNorm1d(out_channels)
        self.Self_2 = SelfONN1d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation,groups=1,bias=True,q=q,dropout=dropout)
        self.Batch_1 = torch.nn.BatchNorm1d(out_channels)
        self.Batch_2 = torch.nn.BatchNorm1d(out_channels)
        self.act_1 = torch.nn.Tanh()
        self.act_2 = torch.nn.Tanh()
        if in_channels!=out_channels:
            self.resample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )
        else:
           self.resample = None 
        
    def forward(self, x):
        input = x
        x = self.act_1(self.Batch_1(self.Self_1(x)))
        x = self.Batch_2(self.Self_2(x))
        if self.resample:
            input = self.resample(input)
        x = self.act_2(input + x)
        return x


class SuperResBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, q=1,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=False,dropout=None):
        super(SuperResBlock, self).__init__()

        self.Super_1 = SuperONN1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,q=q,bias=bias,stride=stride,padding=padding,dilation=dilation,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=dropout)
        self.Batch_1 = torch.nn.BatchNorm1d(out_channels)
        self.Super_2 = SuperONN1d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,q=q,bias=bias,stride=1,padding=padding,dilation=dilation,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=dropout)
        self.Batch_2 = torch.nn.BatchNorm1d(out_channels)
        self.act_1 = torch.nn.Tanh()
        self.act_2 = torch.nn.Tanh()
        if in_channels!=out_channels:
            self.resample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )
        else:
           self.resample = None 
        
    def forward(self, x):
        input = x
        x = self.act_1(self.Batch_1(self.Super_1(x)))
        x = self.Batch_2(self.Super_2(x))
        if self.resample:
            input = self.resample(input)
        x = self.act_2(input + x)
        return x


class Self_MultiResBlock(nn.Module):
    def __init__(self, in_channels, in_features=8, kernel_size=3, stride=1, q=1,bias=True,padding=1,dilation=1,dropout=None):
        super(Self_MultiResBlock,self).__init__() 
        """
        in_channels = in_channels
        out_channels = 3*in_features 
        """
        self.in_channels = in_channels
        self.in_features = in_features
        self.outp_1 = int(self.in_features*1.0)  
        self.outp_2 = int(self.in_features*1.0)  
        self.outp_3 = int(self.in_features*1.0)  
        self.out_res = self.outp_1+self.outp_2+self.outp_3 

        self.residual_layer = self.conv_bn(ch_in=self.in_channels, ch_out=self.out_res,kernel_size=1,stride=1,padding=0,dilation=1,bias=bias,q=1,dropout=None)
        self.conv3x3 = self.conv_bn(ch_in=self.in_channels, ch_out=self.outp_1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,q=q,dropout=dropout)
        self.conv5x5 = self.conv_bn(ch_in=self.outp_1,      ch_out=self.outp_2,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,q=q,dropout=dropout)
        self.conv7x7 = self.conv_bn(ch_in=self.outp_2,      ch_out=self.outp_3,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,q=q,dropout=dropout)
        self.Tanh = nn.Tanh() 
        self.batchnorm_1 = nn.BatchNorm1d(self.out_res)
        self.batchnorm_2 = nn.BatchNorm1d(self.out_res)

    def conv_bn(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, dilation=1, q=1, bias=True, dropout=None):
        return nn.Sequential(
            SelfONN1d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, q=q, bias=True, dropout=dropout),
            nn.BatchNorm1d(ch_out),
            nn.Tanh()
            )

    def forward(self, x):
        res = self.residual_layer(x)
        sbs = self.conv3x3(x)
        obo = self.conv5x5(sbs)
        cbc = self.conv7x7(obo)
        all_t = torch.cat((sbs, obo, cbc), 1)
        all_t_b = self.batchnorm_1(all_t) 
        out = all_t_b.add(res)
        out = self.Tanh(out)
        out = self.batchnorm_2(out)
        return out


class Super_MultiResBlock(nn.Module):
    def __init__(self, in_channels, in_features=8, kernel_size=3, stride=1, q=1,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=False,dropout=None):
        super(Super_MultiResBlock,self).__init__() 
        """
        in_channels = in_channels
        out_channels = 3*in_features 
        """
        self.in_channels = in_channels
        self.in_features = in_features
        self.outp_1 = int(self.in_features*1.0)  
        self.outp_2 = int(self.in_features*1.0)  
        self.outp_3 = int(self.in_features*1.0)  
        self.out_res = self.outp_1+self.outp_2+self.outp_3 

        self.residual_layer = self.conv_bn(ch_in=self.in_channels, ch_out=self.out_res,kernel_size=1,stride=1,padding=0,dilation=1,bias=bias,q=1,learnable=False,max_shift=0,rounded_shifts=False,dropout=None)
        self.conv3x3 = self.conv_bn(ch_in=self.in_channels, ch_out=self.outp_1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,q=q,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=dropout)
        self.conv5x5 = self.conv_bn(ch_in=self.outp_1, ch_out=self.outp_2,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,q=q,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=dropout)
        self.conv7x7 = self.conv_bn(ch_in=self.outp_2, ch_out=self.outp_3,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,q=q,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=dropout)
        self.Tanh = nn.Tanh() 
        self.batchnorm_1 = nn.BatchNorm1d(self.out_res)
        self.batchnorm_2 = nn.BatchNorm1d(self.out_res)

    def conv_bn(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, dilation=1, q=1, bias=True, learnable=True, max_shift=2, rounded_shifts=False, dropout=None):
        return nn.Sequential(
            SuperONN1d(ch_in,ch_out,kernel_size=kernel_size,q=q,bias=bias,stride=stride,padding=padding,dilation=dilation,learnable=learnable,max_shift=max_shift,rounded_shifts=rounded_shifts,dropout=dropout),
            nn.BatchNorm1d(ch_out),
            nn.Tanh()
            )

    def forward(self, x):
        res = self.residual_layer(x)
        sbs = self.conv3x3(x)
        obo = self.conv5x5(sbs)
        cbc = self.conv7x7(obo)
        all_t = torch.cat((sbs, obo, cbc), 1)
        all_t_b = self.batchnorm_1(all_t) 
        out = all_t_b.add(res)
        out = self.Tanh(out)
        out = self.batchnorm_2(out)
        return out

################################## Novel Models (CNN) ##################################

def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv1d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm1d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )


def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm1d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )


def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )
    
    
def conv_block(ch_in, ch_out, kernel_size=3, padding=1, stride=1):
    return (
        nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm1d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )
    
    
class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()
        self.stride = stride
        assert stride in [1,2]
        hidden_dim = ch_in * expand_ratio
        self.use_res_connect = self.stride == 1 and ch_in == ch_out
        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            dwise_conv(hidden_dim, stride=stride),
            conv1x1(hidden_dim, ch_out)
        ])
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)
        
        
class RODNet(nn.Module):
    def __init__(self, input_ch=1, class_num=2):
        super(RODNet, self).__init__()
        self.configs=[
            # t, c, n, s
            [1, 16, 2, 1],
            [6, 24, 3, 2],
            [6, 48, 4, 2],
            [6, 16, 2, 1],
        ]
        self.stem_conv = conv3x3(input_ch, 16, stride=2)

        layers = []
        input_channel = 16
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c
        self.layers = nn.Sequential(*layers)
        self.pool1 = nn.MaxPool1d(4)
        self.conv1 = conv3x3(32, 64, stride=2)
        self.last_conv = conv1x1(64, 1280)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(1280, class_num, bias=True),
                                        nn.LogSigmoid()
                                        )
    def forward(self, x):
        x1 = self.stem_conv(x)
        x2 = self.layers(x1)
        x3 = self.pool1(x1)
        x_cat = torch.cat((x2,x3), 1)
        x4 = self.conv1(x_cat)
        x5 = self.last_conv(x4)
        x6 = self.avg_pool(x5).view(-1, 1280)
        out = self.classifier(x6)
        return out


class ODNet(nn.Module):
    def __init__(self, input_ch=1, class_num=2):
        super(ODNet, self).__init__()
        self.configs=[
            # t, c, n, s
            [1, 16, 2, 1],
            [4, 24, 4, 2],
            [6, 48, 6, 2],
            [6, 64, 6, 2],
            [4, 32, 4, 2],
            [1, 16, 2, 1],
        ]
        self.stem_conv = conv3x3(input_ch, 16, stride=2)

        layers = []
        input_channel = 16
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c
        self.layers = nn.Sequential(*layers)
        self.pool1 = nn.MaxPool1d(4)
        self.pool2 = nn.AvgPool1d(4)
        self.conv1 = conv_block(16, 64, kernel_size=1, padding=1, stride=2)
        self.conv3 = conv_block(16, 64, kernel_size=3, padding=1, stride=2)
        self.conv5 = conv_block(16, 64, kernel_size=5, padding=1, stride=2)
        self.conv7 = conv_block(16, 64, kernel_size=7, padding=1, stride=2)
        self.last_conv = conv1x1(64, 256)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(256, class_num, bias=True),
                                        nn.LogSigmoid()
                                        )
    def forward(self, x):
        x1 = self.stem_conv(x)
        x2 = self.layers(x1)
        x1p1 = self.pool1(x1)
        x1p2 = self.pool2(x1)
        x3 = torch.cat((x2,x1p1,x1p2), -1)
        x2c1 = self.conv1(x2)
        x3c1 = self.conv1(x3)
        x3c3 = self.conv3(x3)
        x3c5 = self.conv5(x3)
        x3c7 = self.conv7(x3)
        x4 = torch.cat((x2c1,x3c1,x3c3,x3c5,x3c7), -1)
        x5 = self.last_conv(x4)
        x6 = self.avg_pool(x5).view(-1, 256)
        out = self.classifier(x6)
        return out
    