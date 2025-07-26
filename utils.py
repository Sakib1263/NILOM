# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import os
import numpy as np
import pandas as pd
import seaborn as sns
# from skimage import io
from sklearn import preprocessing
from scipy import interp
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
# Image manipulations
from PIL import Image
# Timing utility
from timeit import default_timer as timer
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from cgd import CGD


class CreateDataset(Dataset):
    def __init__(self, X, Y, IDs, return_path=False):  
        self.num_samples = X.shape[0]        
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y,dtype=torch.long)   
        self.IDs = np.int16(IDs) 
        self.return_path = return_path

    def normalize(self,x):
        for idx,data in enumerate(x):
            data -= data.min(dim=-1)[0].unsqueeze(-1)
            data /= (data.max(-1)[0]-data.min(-1)[0]).unsqueeze(-1)
            data *= 2
            data -= 1
            # data -= data.mean(dim=-1).unsqueeze(-1) 
            # data = data/data.std()
            x[idx] = data
        # x = torch.cat((x,x,x), dim=0)
        return x     

    def __len__(self): return self.num_samples
    
    def __getitem__(self,idx): 
        X = self.X[idx] 
        X = self.normalize(X) 
        Y = self.Y[idx] 
        if self.return_path:  
            # return X, Y, 'ID_'+str(self.IDs[idx])  
            return X, Y, 'C-'+str(Y.item())+' ID-'+str(self.IDs[idx]) 
        else:
            return X, Y


def to_one_hot(data):
  L_E = preprocessing.LabelEncoder()
  integer_encoded = L_E.fit_transform(data)  
  onehot_encoder = preprocessing.OneHotEncoder()
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  one_hot_encoded_data = onehot_encoder.fit_transform(integer_encoded)
  return one_hot_encoded_data


def to_one_hot_2(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data 
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1) 
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def plot_conf_mat(Ground_Truth_Labels, Predictions, labels):
  confusion_matrix_raw = confusion_matrix(Ground_Truth_Labels, Predictions, normalize=None)
  confusion_matrix_norm = confusion_matrix(Ground_Truth_Labels, Predictions, normalize='true')
  shape = confusion_matrix_raw.shape
  data = np.asarray(confusion_matrix_raw, dtype=int)
  text = np.asarray(confusion_matrix_norm, dtype=float)
  annots = (np.asarray(["{0:.2f} ({1:.0f})".format(text, data) for text, data in zip(text.flatten(), data.flatten())])).reshape(shape[0],shape[1])
  fig = plt.figure(figsize=(len(labels)*3, len(labels)*2))
  sns.heatmap(confusion_matrix_norm, cmap='Blues', annot=annots, fmt='', annot_kws={'fontsize': 16}, xticklabels=labels, yticklabels=labels, vmax=1)
  plt.title('Confusion Matrix', fontsize=24)
  plt.xlabel("Predicted", fontsize=14)
  plt.ylabel("True", fontsize=14)
  return fig


def plot_multiclass_roc(Ground_Truth_Labels, Predictions_Probabilities, categories):
  # Compute ROC curve and Area Under Curve (AUC) for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  true_labels_one_hot_encoded = to_one_hot(Ground_Truth_Labels)
  Predictions_Probabilities_shape = Predictions_Probabilities.shape
  for i in range(Predictions_Probabilities_shape[1]):
      fpr[i], tpr[i], _ = roc_curve(true_labels_one_hot_encoded[:, i], Predictions_Probabilities[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_one_hot_encoded.ravel(), Predictions_Probabilities.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(Predictions_Probabilities_shape[1])]))
  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(Predictions_Probabilities_shape[1]):
      mean_tpr += interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= Predictions_Probabilities_shape[1]

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  fig = plt.figure(figsize=(12, 10))
  plt.plot(fpr["micro"], tpr["micro"],
          label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

  plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

  for i in range(Predictions_Probabilities_shape[1]):
      plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'''.format(categories[i], roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=2)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate', fontsize=16)
  plt.ylabel('True Positive Rate', fontsize=16)
  plt.title('MultiClass ROC Plot with Respective AUC', fontsize=24)
  plt.legend(loc="lower right")
  return fig


def plot_multiclass_precision_recall_curves(Ground_Truth_Labels, Predictions_Probabilities, categories):
  # For each class
  precision = dict()
  recall = dict()
  average_precision = dict()
  true_labels_one_hot_encoded = to_one_hot(Ground_Truth_Labels)
  Predictions_Probabilities_shape = Predictions_Probabilities.shape
  for i in range(Predictions_Probabilities_shape[1]):
      precision[i], recall[i], _ = precision_recall_curve(true_labels_one_hot_encoded[:, i], Predictions_Probabilities[:, i])
      average_precision[i] = average_precision_score(true_labels_one_hot_encoded[:, i], Predictions_Probabilities[:, i])

  # A "micro-average": quantifying score on all classes jointly
  precision["micro"], recall["micro"], _ = precision_recall_curve(true_labels_one_hot_encoded.ravel(), Predictions_Probabilities.ravel())
  average_precision["micro"] = average_precision_score(true_labels_one_hot_encoded, Predictions_Probabilities, average="micro")
  # print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

  from itertools import cycle
  # setup plot details
  colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

  fig = plt.figure(figsize=(12, 10))
  f_scores = np.linspace(0.2, 0.8, num=4)
  lines = []
  labels = []
  for f_score in f_scores:
      x = np.linspace(0.01, 1)
      y = f_score * x / (2 * x - f_score)
      l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
      plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

  lines.append(l)
  labels.append('iso-f1 curves')
  l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
  lines.append(l)
  labels.append('micro-average Precision-recall (area = {0:0.2f})'''.format(average_precision["micro"]))

  for i, color in zip(range(Predictions_Probabilities_shape[1]), colors):
      l, = plt.plot(recall[i], precision[i], color=color, lw=2)
      lines.append(l)
      labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(categories[i], average_precision[i]))

  fig = plt.gcf()
  fig.subplots_adjust(bottom=0.25)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall', fontsize=16)
  plt.ylabel('Precision', fontsize=16)
  plt.title('MultiClass PRC Plot with Respective Precision-Recall AUC', fontsize=24)
  plt.legend(lines, labels, loc=(0, -.3), prop=dict(size=14))
  return fig


def train(model_to_load,
          model,
          class_num,
          lossType,
          criterion,
          optimizer,
          scheduler,
          stop_criteria,
          train_loader,
          valid_loader,
          test_loader,
          save_file_name,
          train_on_gpu,
          history=[],
          max_epochs_stop=5,
          n_epochs=30,
          epochs_prev=0,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    valid_best_acc = 0.0
    # Number of epochs already trained (if using loaded in model weights)
    epochs = epochs_prev
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')
    overall_start = timer()
    # Main loop
    softmax_layer = nn.Softmax(dim=1)
    for epoch in range(n_epochs):
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0
        train_acc = 0
        valid_acc = 0 
        test_acc = 0 
        # Set to training 
        model.train()
        start = timer()
        # Training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for ii, (data, target) in enumerate(pbar):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
            # Clear gradients
            optimizer.zero_grad() 
            output = model(data)  
            # print(output.size()) 
            if output.dim()==3:
                output = output.squeeze(2)
            # Loss and backpropagation of gradients 
            if lossType=='SoftM_CELoss':
                loss = criterion(output, target.squeeze(1)) 
            elif lossType=='SoftM_MSE':
                temp_out = softmax_layer(output)
                temp_target = to_one_hot_2(target, n_dims=class_num).squeeze().to('cuda')  
                loss = criterion(temp_out, temp_target) 
                del temp_out, temp_target
            elif lossType=='MSE': 
                temp_target = to_one_hot_2(target, n_dims=class_num).squeeze().to('cuda')  
                loss = criterion(output, temp_target)  
                del temp_target 
            loss.backward()  
            # Update the parameters
            optimizer.step() 
            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0) 
            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
            # Track training progress
            # print( 
            #     f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
            #     end='\r') 
            # release memeory (delete variables)
            del output, data, target 
            del loss, accuracy, pred, correct_tensor 
        # After training loops ends, start validation
        epochs += 1
        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()
            # Validation loop
            for data, target in valid_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                # Forward pass
                output = model(data)
                if output.dim()==3:
                    output = output.squeeze(2)
                # Validation loss 
                if lossType=='SoftM_CELoss':
                    loss = criterion(output, target.squeeze(1)) 
                elif lossType=='SoftM_MSE':
                    temp_out = softmax_layer(output)
                    temp_target = to_one_hot_2(target, n_dims=class_num).squeeze().to('cuda')  
                    loss = criterion(temp_out, temp_target) 
                    del temp_out, temp_target
                elif lossType=='MSE': 
                    temp_target = to_one_hot_2(target, n_dims=class_num).squeeze().to('cuda')  
                    loss = criterion(output, temp_target)  
                    del temp_target 
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)
                # Calculate validation accuracy
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred)) 
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * data.size(0)
            # Test loop
            for data, target in test_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                # Forward pass
                output = model(data)
                if output.dim()==3:
                    output = output.squeeze(2)
                # Validation loss 
                if lossType=='SoftM_CELoss':
                    loss = criterion(output, target.squeeze(1)) 
                elif lossType=='SoftM_MSE':
                    temp_out = softmax_layer(output)
                    temp_target = to_one_hot_2(target, n_dims=class_num).squeeze().to('cuda')  
                    loss = criterion(temp_out, temp_target) 
                    del temp_out, temp_target
                elif lossType=='MSE': 
                    temp_target = to_one_hot_2(target, n_dims=class_num).squeeze().to('cuda')  
                    loss = criterion(output, temp_target)  
                    del temp_target 
                # Multiply average loss times the number of examples in batch
                test_loss += loss.item() * data.size(0)
                # Calculate validation accuracy
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred)) 
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                test_acc += accuracy.item() * data.size(0) 
            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            test_loss = test_loss / len(test_loader.dataset)
            # optimizer.setLR(np.float64(train_loss))  
            # Calculate average accuracy
            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(valid_loader.dataset)
            test_acc = test_acc / len(test_loader.dataset)
            history.append([train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc]) 
            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \tTest Loss: {test_loss:.4f}')
                print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%\t Test Accuracy: {100 * test_acc:.2f}%') 
            # release memeory (delete variables)
            del output, data, target
            del loss, accuracy, pred, correct_tensor
            if stop_criteria == 'loss':
                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    checkpoint = { 
                        'epoch': epochs,
                        'loss': train_loss,
                        'accuracy': train_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }
                    torch.save(checkpoint, save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc 
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100*valid_best_acc:.2f}%')
                        total_time = timer() - overall_start
                        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')
                        # Load the best state dict
                        checkpoint = torch.load(save_file_name)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        # Attach the optimizer
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc'
                            ])
                        return model, history
            elif stop_criteria == 'accuracy':
                # Save the model if validation loss decreases
                if valid_acc > valid_best_acc:  
                    # Save model
                    checkpoint = { 
                        'epoch': epochs,
                        'loss': train_loss,
                        'accuracy': train_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }
                    torch.save(checkpoint, save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc 
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100*valid_best_acc:.2f}%')
                        total_time = timer() - overall_start
                        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')
                        # Load the best state dict
                        checkpoint = torch.load(save_file_name)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        # Attach the optimizer
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc']
                            )
                        return model, history
        # Update Scheduler
        scheduler.step(valid_acc) 
    # Load the best state dict
    best_model = torch.load(save_file_name)
    model.load_state_dict(best_model['model_state_dict'])
    # Attach the optimizer 
    optimizer.load_state_dict(best_model['optimizer_state_dict'])
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100*valid_best_acc:.2f}%')
    print(f'{total_time:.2f} Total seconds elapsed. {total_time/(epoch):.2f} seconds per epoch.')
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'val_loss', 'test_loss', 'train_acc','val_acc', 'test_acc'])  
    return model, history
