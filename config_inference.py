# CNN test configuration file 

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######
config['parentdir'] = '/content/'     # main directory
config['Data_file'] = '/content/Data_New'   # data path 
config['batch_size']  = 4                         # batch size, Change to fit hardware
config['lossType'] = 'SoftM_MSE'                         # loss function: 'SoftM_CELoss' or 'SoftM_MSE' or 'MSE' 
config['num_folds']  = 5                           # number of cross validation folds
config['CI']  = 0.9                                # Confidence interval (missied cases with probability>=CI will be reported in excel file)
# config['load_model'] = config['parentdir'] + 'load_model/K_Unet_CXR.pt'    # specify full path of pretrained model pt file 
config['load_model'] = False                       # or set to False to load trained model by train code 
config['labeled_Data'] = True                      # set to true if you have the labeled test set
config['old_name'] = 'SelfAttentionResNet18_q5'         # name of trained model .pt file, same name used in train code
config['new_name'] = 'SelfAttentionResNet18_q5'         # either use 'model_name' or specify a new folder name to save test results, 
config['fold_to_run'] = [1,5]
##################  

##################
config['Results_path'] = '/content/gdrive/MyDrive/Carotid_Artery_Task/1D_classification_pipeline'                             # main results file
config['load_path'] = config['Results_path'] +'/'+ config['old_name']    # load path
config['save_path'] = config['Results_path'] +'/'+ config['new_name']    # new save path 
##################
