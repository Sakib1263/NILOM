# configuration file 
config = {}
config['parentdir'] = ''                          # main directory
config['Data_file'] = 'Data'                      # data path 
config['val_size'] = 0.1                          # how mcuh percentage of train set you want to move in validation set  
config['q_order'] = 5                             # qth order Maclaurin approximation, common values: {1,3,5,7,9}. q=1 is equivalent to conventional CNN
config['batch_size'] = 4                          # batch size, Change to fit hardware
config['lossType'] = 'SoftM_MSE'                  # loss function: 'SoftM_CELoss' or 'SoftM_MSE' or 'MSE' 
config['optim_fc'] = 'Adam'                       # 'Adam' or 'SGD'
config['lr'] = 1e-4                               # learning rate 
config['n_epochs']  = 300                         # number of training epochs
config['epochs_patience'] = 10                    # if val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
config['lr_factor'] = 0.1  
config['max_epochs_stop'] = 20                   # maximum number of epochs with no improvement in validation loss for early stopping
config['num_folds']  = 5                         # number of cross validation folds
# config['load_model'] = config['parentdir'] + 'load_model/SelfONN_trial3_fold_1.pt' # specify path of pretrained model wieghts or set to False to train from scratch      
config['load_model'] = False                      # specify path of pretrained model wieghts or set to False to train from scratch   
config['model_to_load'] = 'ResNet'   # chosse one of the following models: 'CNN_1' 'CNN_2' 'CNN_2' 'CNN_3' 'SelfResNet18'  'ResNet'
config['model_name'] = 'ResNet_PCG_Classification'     # choose a unique name for result folder            
config['fold_to_run'] = [1,1]                     # define as [] to loop through all folds, or specify start and end folds i.e. [3 5]
config['Results_path'] = 'Results'                             # main results file
config['save_path'] = config['Results_path'] +'/'+ config['model_name']              # save path 
