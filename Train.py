# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import h5py
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import os
import numpy as np
import configparser
from os import path
from importlib import import_module
from scipy.io import loadmat
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# customized functions
import pickle
from cgd import CGD
from models import get_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from utils import *


# Parse command line arguments
config_file = configparser.ConfigParser()
config_file.read('Config_Train.ini')

if __name__ ==  '__main__':
    # Training Configurations
    parentdir = config_file["TRAIN"]["parentdir"]              # Root or Parent Directory
    datafile = config_file["TRAIN"]["datafile"]                # Folder containing the dataset
    val_size = float(config_file["TRAIN"]["val_size"])         # Validation percentage for splitting
    q_order = int(config_file["TRAIN"]["q_order"])             # q-order for the Self-ONN or Super-ONN Models
    batch_size = int(config_file["TRAIN"]["batch_size"])       # Batch Size, Change to fit hardware
    lossType = config_file["TRAIN"]["lossType"]                # loss function: 'SoftM_CELoss' or 'SoftM_MSE' or 'MSE'
    optim_fc = config_file["TRAIN"]["optim_fc"]                # 'Adam' or 'SGD'
    lr = float(config_file["TRAIN"]["lr"])                     # learning rate
    stop_criteria = config_file["TRAIN"]["stop_criteria"]      # Stopping criteria: 'loss' or 'accuracy'
    n_epochs = int(config_file["TRAIN"]["n_epochs"])           # number of training epochs
    epochs_patience = int(config_file["TRAIN"]["epochs_patience"])  # if val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
    lr_factor = float(config_file["TRAIN"]["lr_factor"])            # lr_factor, if val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
    max_epochs_stop = int(config_file["TRAIN"]["max_epochs_stop"])  # maximum number of epochs with no improvement in validation loss for early stopping
    num_folds = int(config_file["TRAIN"]["num_folds"])              # number of cross validation folds
    load_model = config_file["TRAIN"].getboolean("load_model")      # load_model: True or False
    load_model_path = config_file["TRAIN"]["load_model_path"]      # specify path of pretrained model wieghts or set to False to train from scratch 
    model_to_load = config_file["TRAIN"]["model_to_load"]      # choose one of the following models: 'CNN_1' 'CNN_2' 'CNN_2' 'CNN_3' 'SelfResNet18' 'ResNet'
    model_name = config_file["TRAIN"]["model_name"]            # choose a unique name for result folder
    aux_logits = config_file["TRAIN"].getboolean("aux_logits")     # Required for models with auxilliary outputs (e.g., InceptionV3)  
    results_path = config_file["TRAIN"]["results_path"]        # main results folder
    fold_start = int(config_file["TRAIN"]["fold_start"])       # The starting fold for training
    fold_last = int(config_file["TRAIN"]["fold_last"])         # The last fold for training
    assert (val_size < 1.0) and (val_size >= 0.0), "Error: Validation Set percentage should be smaller than 1.0 and non-negative!"
    # Create Directory 
    if not path.exists(results_path):
        os.makedirs(results_path)
    # Whether to train on a GPU
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')
    # Number of GPUs
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} GPUs detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False 
    # Select start and end folds
    loop_start = fold_start
    loop_end = fold_last + 1
    # loop through folds
    counter = 0
    for fold_idx in range(loop_start,loop_end):
        if counter == 0:
            print('Training using ' + model_to_load + ' network')
        print(f'Starting training with Fold {fold_idx}')
        save_path = results_path + '/' + model_name + '/' + f'Fold_{fold_idx}'
        if not path.exists(save_path):
            os.makedirs(save_path)
        save_file_name = save_path + '/' + f'{model_name}_fold_{fold_idx}.pt'
        checkpoint_name = save_path + f'/checkpoint.pt'
        # load Data
        if os.path.exists(os.path.join(datafile, f'Data_Fold_{fold_idx}.mat')):
            # data = h5py.File(os.path.join(datafile, f'Data_Fold_{fold_idx}.mat'), 'r')
            data = loadmat(os.path.join(datafile, f'Data_Fold_{fold_idx}.mat'))
        elif os.path.exists(os.path.join(datafile, f'Data_Fold_{fold_idx}.h5')): 
            data = h5py.File(os.path.join(datafile, f'Data_Fold_{fold_idx}.h5'), 'r')
        elif os.path.exists(os.path.join(datafile, f'Data_Fold_{fold_idx}.pickle')):
            filepath = open(os.path.join(datafile, f'Data_Fold_{fold_idx}.pickle'), 'rb')
            data = pickle.load(filepath)
        # Extract Data: Train - Test - Validation
        if val_size == None or val_size == '' or val_size == 0.0:
            X_Train = data['X_Train']
            Y_Train = data['Y_Train']
            X_Val = data['X_Val']
            Y_Val = data['Y_Val']
            assert (X_Train.ndim > 1) and (X_Train.ndim < 4), "Error: Check X_Train Dimensions!"
            assert (Y_Train.ndim > 0) and (Y_Train.ndim < 3), "Error: Check Y_Train Dimensions!"
            assert (X_Val.ndim > 1) and (X_Val.ndim < 4), "Error: Check X_Val Dimensions!"
            assert (Y_Val.ndim > 0) and (Y_Val.ndim < 3), "Error: Check Y_Val Dimensions!"
            if X_Train.ndim == 2:
                X_Train = np.expand_dims(X_Train, axis=1)
            if Y_Train.ndim == 1:
                Y_Train = np.int_(np.expand_dims(Y_Train, axis=1))
            if X_Val.ndim == 2:
                X_Val = np.expand_dims(X_Val, axis=1)
            if Y_Val.ndim == 1:
                Y_Val = np.int_(np.expand_dims(Y_Val, axis=1))
            X_Train_Shape = X_Train.shape
            Y_Train_Shape = Y_Train.shape
            assert (X_Train_Shape[0] == Y_Train_Shape[0]), "Error: Train Data and label samples are mismatching!"
            X_Val_Shape = X_Val.shape
            Y_Val_Shape = Y_Val.shape
            assert (X_Val_Shape[0] == Y_Val_Shape[0]), "Error: Validation Data and label samples are mismatching!"
        elif val_size > 0.0 and val_size < 1.0:
            X = data['X_Train']
            Y = data['Y_Train']
            assert (X.ndim > 1) and (X.ndim < 4), "Error: Check X_Train Dimensions!"
            assert (Y.ndim > 0) and (Y.ndim < 3), "Error: Check Y_Train Dimensions!"
            if X.ndim == 2:
                X = np.expand_dims(X, axis=1)
            if Y.ndim == 1:
                Y = np.int_(np.expand_dims(Y, axis=1))
            X_Train_Shape = X.shape
            Y_Train_Shape = Y.shape
            assert (X_Train_Shape[0] == Y_Train_Shape[0]), "Error: Train Data and label samples are mismatching!"
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X, Y, test_size=0.1, random_state=42)
            X_Val_Shape = X_Val.shape
            Y_Val_Shape = Y_Val.shape
            del X, Y 
        X_Test = data['X_Test']
        Y_Test = data['Y_Test']
        assert (X_Test.ndim > 1) and (X_Test.ndim < 4), "Error: Check X_Test Dimensions!"
        assert (Y_Test.ndim > 0) and (Y_Test.ndim < 3), "Error: Check Y_Test Dimensions!"
        if X_Test.ndim == 2:
            X_Test = np.expand_dims(X_Test, axis=1)
        if Y_Test.ndim == 1:
            Y_Test = np.int_(np.expand_dims(Y_Test, axis=1))
        X_Test_Shape = X_Test.shape
        Y_Test_Shape = Y_Test.shape
        assert (X_Test_Shape[0] == Y_Test_Shape[0]), "Error: Test Data and label samples are mismatching!"
        del data
        print(f'Shape of Train Data: {X_Train_Shape}')
        print(f'Shape of Train Label: {Y_Train_Shape}')
        print(f'Shape of Validation Data: {X_Val_Shape}')
        print(f'Shape of Validation Label: {Y_Val_Shape}')
        print(f'Shape of Test Data: {X_Test_Shape}')
        print(f'Shape of Test Label: {Y_Test_Shape}')
        classes = (np.unique(np.concatenate((Y_Train, Y_Test, Y_Val)), axis=0)).ravel()
        print(f'Classes: {classes}')
        num_channels = X_Train.shape[1]
        # return class name, and create class_to_idx and idx_to_class
        class_num = len(classes)
        print(f'No. of Classes: {class_num}')
        categories = []
        class_to_idx = {}
        idx_to_class = {}
        for class_idx in range(class_num): 
            class_name = classes[class_idx]
            class_to_idx[class_name] = class_idx
            idx_to_class[class_idx] = class_name  
            categories.append(class_name) 
        classes = categories
        del categories
        # train, val, and test dataloaders 
        train_ds = CreateDataset(X=X_Train, Y=Y_Train, IDs=np.arange(len(X_Train)), return_path=False)  
        val_ds = CreateDataset(X=X_Val, Y=Y_Val, IDs=np.arange(len(X_Val)), return_path=False)    
        test_ds = CreateDataset(X=X_Test, Y=Y_Test, IDs=np.arange(len(X_Test)), return_path=False)  
        if (len(train_ds)/batch_size)==0:
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False) 
        else:
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)    
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        test_dl = DataLoader(test_ds, batch_size=batch_size) 
        # load model
        if load_model:
            checkpoint = torch.load(load_model_path)
            model = checkpoint['model']  
            del checkpoint
        else: 
            model = get_model(num_channels,q_order,model_to_load,class_num,train_on_gpu,multi_gpu)
        # check if model on cuda
        if next(model.parameters()).is_cuda:
            print('model device: cuda') 
        # Choose model loss function and optimizer
        # Optimizers
        if optim_fc == 'Adagrad':  
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        elif optim_fc == 'Adam':  
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
        elif optim_fc == 'AdamW':  
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        elif optim_fc == 'Adamax':  
            optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif optim_fc == 'NAdam':  
            optimizer = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
        elif optim_fc == 'RAdam':  
            optimizer = torch.optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif optim_fc == 'RMSprop':  
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        elif optim_fc == 'Rprop':  
            optimizer = torch.optim.Rprop(model.parameters(), lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        elif optim_fc == 'SGD': 
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False)
        else:
            raise ValueError('The pipeline does not support this optimizer. Choose a valid optimizer function from here: https://pytorch.org/docs/stable/optim.html')
        # Loss Functions
        if lossType == 'SoftM_CELoss':
            criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
        elif lossType == 'SoftM_MSE':
           criterion = torch.nn.MSELoss()
        elif lossType == 'MSE':
           criterion = torch.nn.MSELoss() 
        elif lossType == 'NLL':
            criterion = torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean')
        elif lossType == 'MML':
            criterion = torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
        else:
            raise ValueError('Choose a valid loss function from here: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer')
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=epochs_patience, verbose=False, 
            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08) 
        # Training
        model, history = train(
            model_to_load,
            model,
            class_num,
            lossType,
            criterion,
            optimizer,
            scheduler,
            stop_criteria,
            train_dl,
            val_dl,
            test_dl, 
            checkpoint_name,
            train_on_gpu,
            history=[],
            max_epochs_stop=max_epochs_stop,
            n_epochs=n_epochs,
            print_every=1) 
        # Save Trained Model
        TrainChPoint = {} 
        TrainChPoint['model']=model                              
        TrainChPoint['history']=history
        torch.save(TrainChPoint, save_file_name)
        # Training Results
        # We can inspect the training progress by looking at the `history`. 
        # plot loss
        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'val_loss', 'test_loss']: 
            plt.plot(
                history[c], label=c) 
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(save_path + f'/LossPerEpoch_fold_{fold_idx}.png')
        # plt.show()
        # plot accuracy
        plt.figure(figsize=(8, 6))
        for c in ['train_acc', 'val_acc','test_acc']: 
            plt.plot(
                100 * history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch') 
        plt.ylabel('Accuracy')
        plt.savefig(save_path + f'/AccuracyPerEpoch_fold_{fold_idx}.png')
        # plt.show()

        # release memeory (delete variables)
        del  optimizer, scheduler
        del  train_ds, train_dl, val_ds, val_dl
        del  TrainChPoint 
        torch.cuda.empty_cache()

        test_ds = CreateDataset(X=X_Test, Y=Y_Test, IDs=np.arange(len(X_Test)), return_path=True)              
        test_dl = DataLoader(test_ds,batch_size=batch_size) 

        # # Test Accuracy
        all_paths =list()
        test_acc = 0.0
        test_loss = 0.0
        i=0 
        model.eval() 
        # new
        softmax_layer = nn.Softmax(dim=1) 
        # new
        for data, targets, ID in test_dl:
            # Tensors to gpu
            if train_on_gpu:
                data = data.to('cuda', non_blocking=True)
                targets = targets.to('cuda', non_blocking=True)
            # all_targets = torch.cat([all_targets ,targets.numpy()])
            # Raw model output
            out = model(data)
            if out.dim()==3:
                out = out.squeeze(2) 
            if lossType=='SoftM_CELoss':
                loss = criterion(out, targets.squeeze()) 
            elif lossType=='SoftM_MSE':
                temp_out = softmax_layer(out) 
                temp_target = to_one_hot_2(targets, n_dims=class_num).squeeze().to('cuda')  
                loss = criterion(temp_out, temp_target) 
                del temp_out, temp_target
            elif lossType=='MSE': 
                temp_target = to_one_hot_2(targets, n_dims=class_num).squeeze().to('cuda')  
                loss = criterion(out, temp_target)  
                del temp_target 
            test_loss += loss.item() * data.size(0)
            # out = torch.exp(out)
            # new
            out = softmax_layer(out)
            # new
            # pred_probs = torch.cat([pred_probs ,out])
            all_paths.extend(ID)
            targets = targets.cpu()
            if i==0:
                all_targets = targets.numpy()
                pred_probs = out.cpu().detach().numpy()
                # pred_probs
            else: 
                all_targets = np.concatenate((all_targets  ,targets.numpy()))
                pred_probs = np.concatenate((pred_probs  ,out.cpu().detach().numpy()))
            _, temp_label = torch.max(out.cpu(), dim=1)
            correct_tensor = temp_label.eq(targets.data.view_as(temp_label))      # this lin is temporary 
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))   # this lin is temporary 
            test_acc += accuracy.item() * data.size(0)                      # this lin is temporary 
            temp_label = temp_label.detach().numpy()
            if i==0:
                pred_label = temp_label
            else:
                pred_label = np.concatenate((pred_label  ,temp_label))

            i +=1 
        test_loss = test_loss / len(test_dl.dataset)
        test_loss = round(test_loss,4)
        test_acc = test_acc / len(test_dl.dataset)                          # this lin is temporary
        test_acc = round(test_acc*100,2)
        print(f'Test Loss: {test_loss},  Test Accuracy: {test_acc}%')
        # main confusion matrix 
        # new
        all_targets = all_targets.reshape(all_targets.shape[0],)
        # new
        class_index  = list(range(0,class_num)) 
        cm = confusion_matrix(y_true=all_targets, y_pred=pred_label,labels=class_index) 
        cm_per_class = multilabel_confusion_matrix(all_targets, pred_label) 
        # new
        actual_classes = np.unique(all_targets)
        missed_classes = 1*(~np.in1d(class_index, actual_classes)) 
        if np.sum(missed_classes) > 0:
            temp_cm_per_class = np.zeros((class_num,2,2))
            actuatl_class_idx = 0
            for class_idx in range(0,class_num):
                # pass
                if missed_classes[class_idx]==0:
                    temp_cm_per_class[class_idx,:,:] =  cm_per_class[actuatl_class_idx,:,:]
                    actuatl_class_idx += 1
                else:
                    temp_cm_per_class[class_idx,:,:] = np.zeros((2,2)) 
            cm_per_class = temp_cm_per_class
        # new
        n_Class_test = np.sum(cm,1)
        # Saving Test Results
        save_file_name = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
        TestChPoint = {}  
        # TestChPoint['classes']=classes 
        categories = classes
        TestChPoint['categories']=categories
        TestChPoint['class_to_idx']=class_to_idx
        TestChPoint['idx_to_class']=idx_to_class
        TestChPoint['Train_history']=history 
        TestChPoint['n_Class_test']=n_Class_test
        TestChPoint['targets']=all_targets
        TestChPoint['prediction_label']=pred_label
        TestChPoint['prediction_probs']=pred_probs
        TestChPoint['signal_IDs']=all_paths  
        TestChPoint['cm']=cm
        TestChPoint['cm_per_class']=cm_per_class
        torch.save(TestChPoint, save_file_name)
        # torch.load(save_file_name) 
        # release memeory (delete variables)
        del model, criterion, history, test_ds, test_dl
        del data, targets, out, temp_label, 
        del test_acc, test_loss, loss
        del pred_probs, pred_label, all_targets, all_paths, 
        del cm, cm_per_class, TestChPoint
        torch.cuda.empty_cache()
        print(f'completed fold {fold_idx}')
        counter = counter + 1
    print('#############################################################')
    # delete checkpoint 
    # os.remove(checkpoint_name)
    # print("Checkpoint File Removed!")
    # Overall Test results
    load_path = results_path + '/' + model_name
    cumulative_cm = []
    for fold_idx in range(loop_start, loop_end):  
        fold_path = load_path + '/' + f'Fold_{fold_idx}' + '/' f'{model_name}_test_fold_{fold_idx}.pt'
        TestChPoint = torch.load(fold_path)
        if fold_idx==1:
            cumulative_cm = TestChPoint['cm'] 
        else:
            cumulative_cm += TestChPoint['cm']
    Overall_Accuracy = np.sum(np.diagonal(cumulative_cm)) / np.sum(cumulative_cm)
    Overall_Accuracy = round(Overall_Accuracy*100, 2)
    print('Cummulative Confusion Matrix')
    print(cumulative_cm)
    print(f'Overall Test Accuracy: {Overall_Accuracy}')
    print('#############################################################')
