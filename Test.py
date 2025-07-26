# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import h5py
import torch
import torch.nn as nn
from torchvision import transforms
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader 
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import os
import numpy as np
import pandas as pd
import configparser
import pickle
import openpyxl
from os import path
from importlib import import_module
from scipy.io import loadmat, savemat
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
# Image manipulations
from PIL import Image
# customized functions 
from utils import *


def Generate_CSV(model, Data_Loader, save_path, idx_to_class, aux_logits, dataset='Train'):
    # Generate Train CSV 
    all_paths =list()
    i=0
    for data, targets, im_path in Data_Loader:
        # Tensors to gpu
        if train_on_gpu:
            data = data.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking=True)
        out = model(data)
        if aux_logits == False:
            output = torch.exp(out)
        elif aux_logits == True:
            output = torch.exp(out[0])
        # images names
        all_paths.extend(im_path)
        targets = targets.cpu()
        if i==0:
            all_targets = targets.numpy()
            pred_probs = output.cpu().detach().numpy()
        else:
            all_targets = np.concatenate((all_targets, targets.numpy()))
            pred_probs = np.concatenate((pred_probs, output.cpu().detach().numpy()))
        _, temp_label = torch.max(output.cpu(), dim=1)
        temp_label = temp_label.detach().numpy()
        if i==0:
            pred_label = temp_label
        else:
            pred_label = np.concatenate((pred_label, temp_label))
        i +=1
    # Generate CSV from pandas dataframe
    output_csv = pd.DataFrame([])
    output_csv['signal_ID'] = all_paths
    output_csv['target'] = all_targets
    output_csv['pred'] = pred_label
    for prob_column in range(pred_probs.shape[1]):
        output_csv[idx_to_class[prob_column]] = pred_probs[:,prob_column]
    output_csv.to_csv(save_path + f'/{model_name}_output_{dataset}_prob_fold_{fold_idx}.csv')
    print(f'{dataset} Dataframe Write to CSV - Done')
    del model
    del data, targets, out, output, temp_label  
    del pred_probs, pred_label, all_targets, all_paths


def Generate_UnlabeledData_CSV(model, Data_Loader, save_path, idx_to_class, aux_logits, dataset='Train'):
    all_paths =list()
    i=0
    for data, _, im_path in Data_Loader: 
        # Tensors to gpu
        if train_on_gpu:
            data = data.to('cuda', non_blocking=True)
        # model output
        out = model(data)
        if aux_logits == False:
            output = torch.exp(out)
        elif aux_logits == True:
            output = torch.exp(out[0])
        # images names
        all_paths.extend(im_path)
        if i==0:
            pred_probs = output.cpu().detach().numpy()
        else:
            pred_probs = np.concatenate((pred_probs, output.cpu().detach().numpy()))
        _, temp_label = torch.max(output.cpu(), dim=1)
        temp_label = temp_label.detach().numpy()
        if i==0:
            pred_label = temp_label
        else:
            pred_label = np.concatenate((pred_label, temp_label))
        i +=1

    output_csv = pd.DataFrame([])
    output_csv['signal_ID'] = all_paths
    output_csv['pred'] = pred_label
    
    for prob_column in range(pred_probs.shape[1]):
        output_csv[idx_to_class[prob_column]] = pred_probs[:,prob_column]
    output_csv.to_csv(save_path + f'/{model_name}_output_{dataset}_prob_fold_{fold_idx}.csv')
    print(f'{dataset} Dataframe Write to CSV - Done')
    del model
    del data, targets, out, output, temp_label  
    del pred_probs, pred_label, all_paths

 
# Parse command line arguments
config_file = configparser.ConfigParser()
config_file.read('Config_Test.ini')

if __name__ ==  '__main__':
    # Test Configurations
    parentdir = config_file["TEST"]["parentdir"]                  # Root or Parent Directory
    datafile = config_file["TEST"]["datafile"]                    # Folder containing the dataset
    batch_size = int(config_file["TEST"]["batch_size"])           # Batch Size, Change to fit hardware
    lossType = config_file["TEST"]["lossType"]                    # loss function: 'SoftM_CELoss' or 'SoftM_MSE' or 'MSE'
    num_folds = int(config_file["TEST"]["num_folds"])             # number of cross validation folds
    CI = float(config_file["TEST"]["CI"])                         # number of cross validation folds
    load_model = config_file["TEST"].getboolean("load_model")     # load_model: True or False
    load_model_path = config_file["TEST"]["load_model_path"]      # specify path of pretrained model wieghts or set to False to train from scratch 
    labeled_data = config_file["TEST"].getboolean("labeled_data") # set to true if you have the labeled test set
    model_name = config_file["TEST"]["model_name"]                # choose a unique name for result folder
    aux_logits = config_file["TEST"].getboolean("aux_logits")     # Required for models with auxilliary outputs (e.g., InceptionV3)  
    fold_start = int(config_file["TEST"]["fold_start"])           # The starting fold for training
    fold_last = int(config_file["TEST"]["fold_last"])             # The last fold for training
    N_steps = int(config_file["TEST"]["N_steps"])                 # The last fold for training
    results_path = config_file["TEST"]["results_path"]            # main results folder
    # Create  Directory
    if not path.exists(results_path):  
        os.makedirs(results_path)
    # Whether to train on a GPU
    train_on_gpu = cuda.is_available()
    print(f'Train on GPU: {train_on_gpu}')
    # Number of gpus
    if train_on_gpu: 
        gpu_count = cuda.device_count()
        print(f'{gpu_count} GPUs detected.')
        if gpu_count > 1:
            multi_gpu = True 
        else:
            multi_gpu = False 
    test_history = []
    index = []
    # Select start and end folds
    loop_start = fold_start
    loop_end = fold_last + 1
    print(f'Combined Evaluation of Folds {fold_start} to {fold_last}...')
    for fold_idx in range(loop_start, loop_end): 
        print('#############################################################')
        print(f'Started fold {fold_idx}')
        load_path = results_path + '/' + model_name + '/' + f'Fold_{fold_idx}'
        if not path.exists(load_path):  
            os.makedirs(load_path)
        load_file = load_path + '/' + f'{model_name}_test_fold_{fold_idx}.pt'
        if os.path.exists(os.path.join(datafile, f'Data_Fold_{fold_idx}.mat')): 
            data = loadmat(os.path.join(datafile, f'Data_Fold_{fold_idx}.mat'))
        elif os.path.exists(os.path.join(datafile, f'Data_Fold_{fold_idx}.h5')): 
            data = h5py.File(os.path.join(f'Data_Fold_{fold_idx}.h5'), 'r')
        elif os.path.exists(os.path.join(datafile, f'Data_Fold_{fold_idx}.pickle')):
            filepath = open(os.path.join(datafile, f'Data_Fold_{fold_idx}.pickle'), 'rb')
            data = pickle.load(filepath)
        X_Test = data['X_Test']
        assert (X_Test.ndim > 1) and (X_Test.ndim < 4), "Error: Check X_Test Dimensions!"
        if X_Test.ndim == 2:
            X_Test = np.expand_dims(X_Test, axis=1)
        Y_Test = data['Y_Test']
        assert (Y_Test.ndim > 0) and (Y_Test.ndim < 3), "Error: Check Y_Test Dimensions!"
        if Y_Test.ndim == 1:
            Y_Test = np.int_(np.expand_dims(Y_Test, axis=1))
        X_Test_Shape = X_Test.shape
        Y_Test_Shape = Y_Test.shape
        assert (X_Test_Shape[0] == Y_Test_Shape[0]), "Error: Data and label samples are mismatching!"
        classes = (np.unique(Y_Test)).ravel()
        LoadChPoint = torch.load(load_file)
        categories = LoadChPoint['categories']
        class_to_idx = LoadChPoint['class_to_idx']
        idx_to_class = LoadChPoint['idx_to_class']
        n_Class_test = LoadChPoint['n_Class_test'] 
        # return class name, and create class_to_idx and idx_to_class
        class_num = len(classes)
        # test dataloaders     
        test_ds = CreateDataset(X=X_Test, Y=Y_Test, IDs=np.arange(len(X_Test)), return_path=True)            
        test_dl = DataLoader(test_ds, batch_size=batch_size) 
        # load model 
        if load_model:
            checkpoint = torch.load(load_model_path)
            model = checkpoint['model']  
            del checkpoint 
        else: 
            pt_file = load_path + '/' f'{model_name}_fold_{fold_idx}.pt'
            checkpoint = torch.load(pt_file) 
            model = checkpoint['model'] 
            del   pt_file, checkpoint
        model = model.to('cuda')  
        # check if model on cuda
        if next(model.parameters()).is_cuda:
            print('model device: cuda')
        # Set to evaluation mode
        model.eval()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_params)
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
        # new
        softmax_layer = nn.Softmax(dim=1) 
        # new
        if labeled_data:
            all_paths =list()
            test_acc = 0.0
            test_loss = 0.0
            pbar = tqdm(test_dl, desc=f"Testing")
            for ii, (data, targets, im_path) in enumerate(pbar):
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                    targets = targets.to('cuda', non_blocking=True)
                # model output
                out = model(data)
                if out.dim() == 3:
                    out = out.squeeze(2)
                # loss
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
                # images names
                all_paths.extend(im_path)
                targets = targets.cpu()
                if ii==0:
                    all_targets = targets.numpy()
                    pred_probs = out.cpu().detach().numpy()
                else:
                    all_targets = np.concatenate((all_targets  ,targets.numpy()))
                    pred_probs = np.concatenate((pred_probs  ,out.cpu().detach().numpy()))
                _, temp_label = torch.max(out.cpu(), dim=1)
                correct_tensor = temp_label.eq(targets.data.view_as(temp_label))      # this lin is temporary 
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))         # this lin is temporary 
                test_acc += accuracy.item() * data.size(0)                            # this lin is temporary 
                temp_label = temp_label.detach().numpy()
                if ii==0:
                    pred_label = temp_label
                else:
                    pred_label = np.concatenate((pred_label,temp_label))
            test_loss = test_loss / len(test_dl.dataset) 
            test_loss = round(test_loss,4)
            test_acc = test_acc / len(test_dl.dataset)                          # this lin is temporary
            test_acc = round(test_acc*100,2)
            print(f'Test Loss: {test_loss},  Test Accuracy: {test_acc}%')
            # new
            all_targets = all_targets.reshape(all_targets.shape[0],)
            # new
            from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
            # main confusion matrix
            class_index  = list(range(0,class_num))
            cm = confusion_matrix(y_true=all_targets, y_pred=pred_label,labels=class_index)
            cm_per_class = multilabel_confusion_matrix(all_targets, pred_label)
            # Saving Test Results
            save_file_name = load_path + '/' + f'{model_name}_test_fold_{fold_idx}.pt'
            TestChPoint = {} 
            TestChPoint['categories']=categories
            TestChPoint['class_to_idx']=class_to_idx
            TestChPoint['idx_to_class']=idx_to_class
            TestChPoint['n_Class_test']=n_Class_test
            TestChPoint['targets']=all_targets
            TestChPoint['prediction_label']=pred_label
            TestChPoint['prediction_probs']=pred_probs
            TestChPoint['signal_IDs']=all_paths 
            TestChPoint['cm']=cm
            TestChPoint['cm_per_class']=cm_per_class
            torch.save(TestChPoint, save_file_name)
            print('Generating CSV Files from Individual Predictions...')
            Generate_CSV(model=model, Data_Loader=test_dl, save_path=load_path, idx_to_class=idx_to_class, aux_logits=aux_logits, dataset='Test')
        else:
            all_paths =list()
            for ii, (data, _, im_path) in enumerate(pbar): 
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                # model output
                out = model(data)
                if out.dim()==3:
                    out = out.squeeze(2)
                # out = torch.exp(out)
                # new
                out = softmax_layer(out)
                # new
                # images names
                all_paths.extend(im_path)
                if ii==0:
                    pred_probs = out.cpu().detach().numpy()
                else:
                    pred_probs = np.concatenate((pred_probs  ,out.cpu().detach().numpy()))
                _, temp_label = torch.max(out.cpu(), dim=1)
                temp_label = temp_label.detach().numpy()
                if ii==0:
                    pred_label = temp_label
                else:
                    pred_label = np.concatenate((pred_label  ,temp_label))
            # Saving Test Results
            save_file_name = load_path + '/' + f'{model_name}_test_fold_{fold_idx}.pt'
            TestChPoint = {} 
            TestChPoint['categories']=categories
            TestChPoint['class_to_idx']=class_to_idx
            TestChPoint['idx_to_class']=idx_to_class
            TestChPoint['n_Class_test']=n_Class_test 
            TestChPoint['prediction_label']=pred_label
            TestChPoint['prediction_probs']=pred_probs
            TestChPoint['signal_names']=all_paths 
            torch.save(TestChPoint, save_file_name) 
            print('Generating CSV Files from Individual Predictions...') 
            Generate_UnlabeledData_CSV(model=model, Data_Loader=test_dl, save_path=load_path, idx_to_class=idx_to_class, aux_logits=aux_logits, dataset='Test')
        # Measure Inference Time
        Total_time = 0.0
        for i in range(N_steps):
            input_time = timer()
            out = model(data) 
            output_time = timer() 
            output_time = output_time - input_time
            Total_time = Total_time + output_time
            del out
        Total_time = Total_time/N_steps
        print(f'Total Inference Time: {((Total_time*1000)/N_steps):.4} ms')
       # release memeory (delete variables)
        if labeled_data: 
            del model, criterion, test_ds, test_dl
            del data, targets, temp_label  
            del test_acc, test_loss, loss
            del pred_probs, pred_label, all_targets, all_paths, 
            del cm, cm_per_class, TestChPoint
        else:
            del model, criterion, test_ds, test_dl
            del data, out, temp_label  
            del pred_probs, pred_label, all_paths, 
            del TestChPoint 
        torch.cuda.empty_cache()
        print(f'Completed fold {fold_idx}') 
    # Combined Evaluation of All CV Folds
    print('#############################################################') 
    if labeled_data: 
        all_Missed_c = list() 
        load_path = results_path + '/' + model_name
        for fold_idx in tqdm(range(loop_start, loop_end), desc="Combined Evaluation"):
            # load checkpoint
            fold_path = load_path + '/' + f'Fold_{fold_idx}'
            model_path = fold_path + '/' + f'{model_name}_test_fold_{fold_idx}.pt'
            TestChPoint = torch.load(model_path)
            if fold_idx==loop_start:
                targets = TestChPoint['targets']
                pred = TestChPoint['prediction_label']
                pred_probs = TestChPoint['prediction_probs']
                signal_IDs  = TestChPoint['signal_IDs']
            else:
                targets = np.concatenate([targets, TestChPoint['targets']])
                pred = np.concatenate([pred, TestChPoint['prediction_label']])
                pred_probs = np.concatenate([pred_probs, TestChPoint['prediction_probs']])
                signal_IDs.extend(TestChPoint['signal_IDs'])
            # find missed cases (probs, image path)
            categories = TestChPoint['categories']
            n = len(categories)
            # temp
            current_fold_target = TestChPoint['targets']
            current_fold_pred = TestChPoint['prediction_label']
            current_fold_signal_IDs = TestChPoint['signal_IDs'] 
            current_fold_pred_probs = TestChPoint['prediction_probs'] 
            # missed_idx = np.argwhere(1*(targets==pred) == 0)
            missed_idx = np.argwhere(1*(current_fold_target==current_fold_pred) == 0)
            # temp 
            m = len(missed_idx) 
            missed_probs = np.zeros((m,n)) 
            for i in range(len(missed_idx)):
                index = int(missed_idx [i])
                all_Missed_c.extend([f'fold_{fold_idx}/'+signal_IDs[index]]) 
                missed_probs[i,:] = pred_probs[index,:]
            if fold_idx==loop_start:
                all_missed_p = missed_probs
            else: 
                all_missed_p = np.concatenate((all_missed_p,missed_probs))
            # main confusion matrix
            cm = confusion_matrix(current_fold_target, current_fold_pred)
            # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
            cm_per_class = multilabel_confusion_matrix(current_fold_target, current_fold_pred)
            # Overall Accuracy
            Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
            Overall_Accuracy = round(Overall_Accuracy*100, 2)
            # Create confusion matrix table (pd.DataFrame)
            cm_table = pd.DataFrame(cm, index=categories, columns=categories)
            # Generate Confusion matrix figure
            cm_plot = plot_conf_mat(current_fold_target, current_fold_pred, labels=categories)
            cm_plot.savefig(fold_path + f'/{model_name}_Confusion_Matrix_fold_{fold_idx}.png', dpi=600)
            # Generate Multiclass ROC curve figure
            # roc_plot = plot_multiclass_roc(current_fold_target, current_fold_pred_probs, categories)
            # roc_plot.savefig(fold_path + f'/{model_name}_ROC_plot_fold_{fold_idx}.png', dpi=600)
            # Generate Multiclass precision-recall curve figure
            # prc_plot = plot_multiclass_precision_recall_curves(current_fold_target, current_fold_pred_probs, categories)
            # prc_plot.savefig(fold_path + f'/{model_name}_PRC_plot_fold_{fold_idx}.png', dpi=600)
            Eval_Mat = []
            # Per class metricies
            for i in range(len(categories)):
                TN = cm_per_class[i][0][0] 
                FP = cm_per_class[i][0][1]   
                FN = cm_per_class[i][1][0]  
                TP = cm_per_class[i][1][1]  
                Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                Precision = round(100*(TP)/(TP+FP), 2)  
                Sensitivity = round(100*(TP)/(TP+FN), 2) 
                F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)  
                Specificity = round(100*(TN)/(TN+FP), 2)  
                Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
            # Sizes of each class
            s = np.sum(cm,axis=1) 
            # Create tmep excel table 
            headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
            temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
            # Weighted average of per class metricies
            Accuracy = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2) 
            Precision = round(temp_table['Precision'].dot(s)/np.sum(s), 2)  
            Sensitivity = round(temp_table['Sensitivity'].dot(s)/np.sum(s), 2)  
            F1_score = round(temp_table['F1_score'].dot(s)/np.sum(s), 2)  
            Specificity = round(temp_table['Specificity'].dot(s)/np.sum(s), 2)   
            values = [Accuracy, Precision, Sensitivity, F1_score, Specificity]
            # Create per class metricies excel table with weighted average row
            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
            categories_wa = categories + ['Weighted Average']
            Eval_table = pd.DataFrame(Eval_Mat, index=categories_wa, columns=headers)
            # Create confusion matrix table (pd.DataFrame)
            Overall_Acc = pd.DataFrame(Overall_Accuracy, index=['Overall_Accuracy'] , columns=[' '])
            # Save to excel file   
            new_savepath = fold_path + f'/{model_name}_fold_{fold_idx}.xlsx'  # file to save 
            writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
            # Sheet 1 (Evaluation metricies) + (Commulative Confusion Matrix) 
            col = 1; row = 1 
            Eval_table.to_excel(writer, "Results", startcol=col, startrow=row)
            row = row+7+len(class_to_idx)
            Overall_Acc.to_excel(writer, "Results", startcol=col, startrow=row, header=None)
            col = col+8; row=1   
            Predicted_Class = pd.DataFrame(['Predicted Class'])
            Predicted_Class.to_excel(writer, "Results", startcol=col+1, startrow=row, header=None, index=None)
            row = 2     
            cm_table.to_excel(writer, "Results", startcol=col, startrow=row)
            # save 
            writer.close()
        # find missed cases with high CI (probs, image path)
        temp = np.max(all_missed_p,axis=1)
        temp_idx = np.argwhere(temp >= CI) 
        unsure_missed_c = list() 
        unsure_missed_p = np.zeros((len(temp_idx), n))
        for i in range(len(temp_idx)):
            index = int(temp_idx[i])
            unsure_missed_c.extend([ all_Missed_c[index] ]) 
            unsure_missed_p[i,:] =  all_missed_p[index,:]
        categories = TestChPoint['categories']
        n = len(categories)
        # main confusion matrix
        class_index  = list(range(0,class_num))
        cm = confusion_matrix(y_true=targets, y_pred=pred, labels=class_index)
        # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
        # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP   
        cm_per_class = multilabel_confusion_matrix(targets, pred) 
        # Overall Accuracy
        Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
        Overall_Accuracy = round(Overall_Accuracy*100, 2)
        # create missed and unsure missed tables
        missed_table = pd.DataFrame(all_Missed_c, columns=[f'Missed Cases'])
        unsure_table = pd.DataFrame(unsure_missed_c, columns=[f'Unsure Missed Cases (CI={CI})']) 
        missed_prob_table = pd.DataFrame(np.round(all_missed_p,4), columns=categories) 
        unsure_prob_table = pd.DataFrame(np.round(unsure_missed_p,4), columns=categories) 
        # create confusion matrix table (pd.DataFrame)
        cm_table = pd.DataFrame(cm, index=TestChPoint['categories'] , columns=TestChPoint['categories'])
        # Generate Confusion matrix figure
        cm_plot = plot_conf_mat(targets, pred, labels=categories)
        cm_plot.savefig(load_path + f'/{model_name}_Overall_Confusion_Matrix.png', dpi=600)
        # Generate Multiclass ROC curve figure
        # roc_plot = plot_multiclass_roc(targets, pred_probs, categories)
        # roc_plot.savefig(load_path + f'/{model_name}_Overall_ROC_plot.png', dpi=600)
        # Generate Multiclass precision-recall curve figure
        # prc_plot = plot_multiclass_precision_recall_curves(targets, pred_probs, categories)
        # prc_plot.savefig(load_path + f'/{model_name}_Overall_PRC_plot.png', dpi=600)
        Eval_Mat = [] 
        # per class metricies
        for i in range(len(categories)):
            TN = cm_per_class[i][0][0] 
            FP = cm_per_class[i][0][1]   
            FN = cm_per_class[i][1][0]  
            TP = cm_per_class[i][1][1]  
            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
            if np.isnan(Accuracy):
                Accuracy = 0.0
            Precision = round(100*(TP)/(TP+FP+1e-6), 2)  
            Sensitivity = round(100*(TP)/(TP+FN+1e-6), 2)  
            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity+1e-6), 2)  
            Specificity = round(100*(TN)/(TN+FP+1e-6), 2)  
            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
        # sizes of each class
        s = np.sum(cm,axis=1) 
        # create tmep excel table 
        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
        # weighted average of per class metricies
        Accuracy = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2) 
        Precision = round(temp_table['Precision'].dot(s)/np.sum(s), 2)  
        Sensitivity = round(temp_table['Sensitivity'].dot(s)/np.sum(s), 2)  
        F1_score = round(temp_table['F1_score'].dot(s)/np.sum(s), 2)  
        Specificity = round(temp_table['Specificity'].dot(s)/np.sum(s), 2)   
        values = [Accuracy, Precision, Sensitivity, F1_score, Specificity]
        # create per class metricies excel table with weighted average row
        Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
        categories.extend(['Weighted Average'])
        Eval_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
        # create confusion matrix table (pd.DataFrame)
        Overall_Acc = pd.DataFrame(Overall_Accuracy, index=['Overall_Accuracy'] , columns=[' '])
        # print('Cummulative Confusion Matrix')
        print('\n') 
        print(cm_table) 
        print('\n') 
        # print('Evaluation Matricies')
        print(Eval_table)
        print('\n')  
        # print('Overall Accuracy')
        print(Overall_Acc)
        print('\n') 
        # save to excel file   
        new_savepath = load_path + '/' + f'{model_name}.xlsx'  # file to save 
        writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
        # sheet 1 (Unsure missed cases) + (Evaluation metricies) + (Commulative Confusion Matrix) 
        col =0; row =2 
        unsure_table.to_excel(writer, "Results", startcol=col,startrow=row) 
        col =2; row =2 
        unsure_prob_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
        col =col+class_num+2; row =2 
        Eval_table.to_excel(writer, "Results", startcol=col,startrow=row)
        row = row +7
        Overall_Acc.to_excel(writer, "Results", startcol=col,startrow=row, header=None)
        col = col+8; row=1  
        Predicted_Class = pd.DataFrame(['Predicted Class']) 
        Predicted_Class.to_excel(writer, "Results", startcol=col+1,startrow=row, header=None, index=None)
        row =2     
        cm_table.to_excel(writer, "Results", startcol=col,startrow=row)
        # sheet 2 (All missed cases)
        col =0; row =2 
        missed_table.to_excel(writer, "Extra", startcol=col,startrow=row)
        col =2; row =2 
        missed_prob_table.to_excel(writer, "Extra", startcol=col,startrow=row, index=None)
        # save 
        writer.close()  
        # new 
        # Save needed variables to create ROC curves
        ROC_checkpoint = {} 
        ROC_checkpoint['prediction_label'] = pred
        ROC_checkpoint['prediction_probs'] = pred_probs
        ROC_checkpoint['targets'] = targets
        ROC_checkpoint['class_to_idx']=class_to_idx
        ROC_checkpoint['idx_to_class']=idx_to_class 
        ROC_path_pt = load_path + '/' + f'{model_name}_roc_inputs.pt'  # file to save 
        ROC_path_mat = load_path + '/' + f'{model_name}_roc_inputs.mat'  # file to save 
        torch.save(ROC_checkpoint,ROC_path_pt) 
        # import scipy.io as spio 
        # spio.savemat(ROC_path_mat, ROC_checkpoint) 
        savemat(ROC_path_mat, ROC_checkpoint) 
        # new
    else:
        for fold_idx in range(1, num_folds+1): 
            # load checkpoint
            load_path = results_path + '/' + model_name
            fold_path = load_path + '/' + f'Fold_{fold_idx}' + '/' + f'{model_name}_test_fold_{fold_idx}.pt'
            TestChPoint = torch.load(fold_path)
            #
            categories = TestChPoint['categories']  
            #
            temp_pred = TestChPoint['prediction_label']
            pred_probs = TestChPoint['prediction_probs'] 
            signal_IDs  = TestChPoint['signal_IDs']
            for i in range(len(temp_pred)): 
                if i==0:
                    pred = [ idx_to_class[temp_pred[i]] ] 
                else:
                    pred.extend([ idx_to_class[temp_pred[i]] ])  
            # create missed and unsure missed tables
            input_names_table = pd.DataFrame(signal_IDs, columns=[f'Input Image']) 
            pred_table = pd.DataFrame(pred, columns=[f'Prediction']) 
            prob_table = pd.DataFrame(np.round(pred_probs, 4), columns=categories) 
            # save to excel file   
            new_savepath = fold_path + '/' + f'{model_name}_fold{fold_idx}.xlsx'  # file to save 
            writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
            # sheet 1 (input images) + (predictions) + (predictions probabilities) 
            col =0; row =2 
            input_names_table.to_excel(writer, "Results", startcol=col,startrow=row)
            col =2; row =2  
            pred_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
            col =3; row =2 
            prob_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
            # save 
            writer.close()
    print('#############################################################') 
