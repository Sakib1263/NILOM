# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# SelfONN 
from SelfONN import SuperONN1d, SelfONN1d
from models_utils import *

def get_model(num_channels, q, model_name,class_num,train_on_gpu,multi_gpu):
    if model_name == 'Self_Net':  
        model = torch.nn.Sequential(
                # 1D CNN for feature extraction 
                Self_Net(in_channels=num_channels, features=8, q=q),
                # MLP classifier 
                torch.nn.Linear(in_features=88,out_features=class_num)  
            )  
    elif model_name == 'Self_MobileNet':
        model = Self_MobileNet(input_channel = num_channels, last_layer_channel = 32, class_num= class_num)
    elif model_name == 'Self_DenseMobileNet':
        model = Self_DenseMobileNet(input_channel = num_channels, last_layer_channel = 32, class_num= class_num)
    elif model_name == 'Super_Net':  
        model = torch.nn.Sequential(
                # 1D CNN for feature extraction 
                Super_Net(in_channels=num_channels, features=8, q=q, learnable=True, max_shift=2,rounded_shifts=False),
                # MLP classifier 
                torch.nn.Linear(in_features=88,out_features=class_num)  
            )  
    elif model_name == 'Self_ResNet':  
        model = torch.nn.Sequential(
                # 1D CNN for feature extraction 
                Self_ResNet(in_channels=num_channels, features=8, q=q),
                # MLP classifier 
                torch.nn.Linear(in_features=88,out_features=class_num) 
            )  
    elif model_name == 'Super_ResNet':  
        model = torch.nn.Sequential(
                # 1D CNN for feature extraction 
                Super_ResNet(in_channels=num_channels, features=8, q=q, learnable=True,max_shift=2,rounded_shifts=False),
                # MLP classifier 
                torch.nn.Linear(in_features=88,out_features=class_num)
            )  
    elif model_name == 'Self_MultiResNet':  
        model = torch.nn.Sequential(
                # 1D CNN for feature extraction 
                Self_MultiResNet(in_channels=num_channels, q=q),
                # MLP classifier 
                torch.nn.Linear(in_features=132,out_features=class_num)
            )  
    elif model_name == 'Super_MultiResNet':  
        model = torch.nn.Sequential(
                # 1D CNN for feature extraction 
                Super_MultiResNet(in_channels=num_channels, q=q, learnable=True,max_shift=2,rounded_shifts=False),
                # MLP classifier 
                torch.nn.Linear(in_features=132,out_features=class_num)
            )  
    elif model_name == 'StackCNN':  
        model = StackCNN(in_channels=num_channels, out_channels=class_num, features=8, q=q, learnable=True,max_shift=2,rounded_shifts=False)
    elif model_name == 'RODNet':
        model = RODNet(input_ch=num_channels, class_num=class_num)
    elif model_name == 'ODNet':
        model = ODNet(input_ch=num_channels, class_num=class_num)
################################ old models ############################################
    elif model_name == 'SelfResNet18':
        from SelfResNet import SelfResNet
        model = SelfResNet(input_ch=num_channels, class_num=class_num, q_order=q, in_ft=8, BatchNorm=False) 
    elif model_name == 'SelfResAttentioNet18':
        from SelfResNet import SelfResAttentioNet
        model = SelfResAttentioNet(input_ch=num_channels, class_num=class_num, q_order=q, in_ft=8, BatchNorm=False)
    elif model_name == 'CNN_1':    
        model = torch.nn.Sequential(
            # 1st layer (1D CNN)  
            SelfONN1d(in_channels=num_channels,out_channels=16,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SelfONN1d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(8),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten(),
            # 3rd (Output) layer (MLP) 
            torch.nn.Linear(in_features=88,out_features=class_num),  
        )  
    elif model_name == 'CNN_2':    
        model = torch.nn.Sequential(
            # 1st layer (1D CNN)  
            SelfONN1d(in_channels=num_channels,out_channels=16,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SelfONN1d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(8),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten(),
            # 3rd layer (MLP)
            torch.nn.Linear(in_features=88,out_features=32), 
            torch.nn.BatchNorm1d(32),
            torch.nn.Tanh(),
            # 4th (Output) layer (MLP)  
            torch.nn.Linear(in_features=32,out_features=class_num),  
        )  

    elif model_name == 'CNN_3':    
        model = torch.nn.Sequential(
            # 1st layer (1D CNN)  
            SelfONN1d(in_channels=num_channels,out_channels=16,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(2),
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SelfONN1d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(2),
            torch.nn.BatchNorm1d(16),
            torch.nn.Tanh(),
            # 3rd layer (1D CNN)  
            SelfONN1d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(2),
            torch.nn.BatchNorm1d(8), 
            torch.nn.Tanh(),
            # 4th layer (1D CNN)
            SelfONN1d(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(2), 
            torch.nn.BatchNorm1d(8),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten(),
            # 5th (Output) layer (MLP) 
            torch.nn.Linear(in_features=72,out_features=class_num),  
        )  

    elif model_name == 'CNN_4':    
        model = torch.nn.Sequential(
            # 1st layer (1D CNN)  
            SelfONN1d(in_channels=num_channels,out_channels=16,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(2), 
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SelfONN1d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(2), 
            torch.nn.BatchNorm1d(16),
            torch.nn.Tanh(),
            # 3rd layer (1D CNN)  
            SelfONN1d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(2), 
            torch.nn.BatchNorm1d(8), 
            torch.nn.Tanh(),
            # 4th layer (1D CNN)
            SelfONN1d(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q,mode='fast'), 
            torch.nn.MaxPool1d(2), 
            torch.nn.BatchNorm1d(8),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten(),
            # 5th layer (MLP)
            torch.nn.Linear(in_features=72,out_features=32), 
            torch.nn.BatchNorm1d(32), 
            torch.nn.Tanh(),
            # 6th (Output) layer (MLP)  
            torch.nn.Linear(in_features=32,out_features=class_num),  
        )  
    
    elif model_name == 'SuperONN_1':    
        model = torch.nn.Sequential(
            # 1st layer (1D CNN)  
            SuperONN1d(in_channels=num_channels,out_channels=16,kernel_size=3,q=1,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=False,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(16), 
            torch.nn.Tanh(),
            # 2nd layer (1D CNN)
            SuperONN1d(in_channels=16,out_channels=8,kernel_size=3,q=1,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=False,dropout=None),
            torch.nn.MaxPool1d(4),
            torch.nn.BatchNorm1d(8),
            torch.nn.Tanh(),
            # Flatten layer
            torch.nn.Flatten(),
            # 3rd (Output) layer (MLP) 
            torch.nn.Linear(in_features=88,out_features=class_num),  
        )  


    elif model_name == 'ResNet':
        """  
        Pararmetes:
            n_length: dimention of input (resolution) [16, 64, 256, 1024, 4096]
            in_channels: dim of input, the same as n_channel
            n_block: number of blocks [2, 4, 8, 16]
            base_filters: number of filters in the first several Conv layer, it will double at every 4 layers [8, 16, 32, 64, 128]
            kernel_size: width of kernel [2, 4, 8, 16]
            stride: stride of kernel moving
            groups: set larget to 1 as ResNeXt
            n_classes: number of classes
            ## change the hyper-parameters for your own data
            # recommend: (n_block, downsample_gap, increasefilter_gap) = (8, 1, 2)
        """ 
        from resnet1d import ResNet1D 
        # finetune the first three paarmeters
        n_block = 4       # finetune  [2, 4, 8, 16]        
        base_filters= 8  # finetune  [8, 16, 32, 64, 128] 
        kernel_size = 16  # finetune  [2, 4, 8, 16]
        stride=2
        groups=1 
        downsample_gap=max(n_block//8, 1)
        increasefilter_gap=max(n_block//4, 1)
        model = ResNet1D(in_channels=num_channels, base_filters=base_filters, kernel_size=kernel_size, stride=stride, groups=groups, n_block=n_block, n_classes=class_num, downsample_gap=downsample_gap, increasefilter_gap=increasefilter_gap, use_bn=True, use_do=True, verbose=False)

    # add a new model  
    
    # add a new model 
    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda') 
    if multi_gpu:
        model = nn.DataParallel(model)
    return model 
