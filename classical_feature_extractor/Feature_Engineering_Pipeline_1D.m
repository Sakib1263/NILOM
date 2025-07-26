%% Train Feature Extraction, Concatenation and Saving
warning('off')
% Load Data for Feature Extraction
clc;
clear;
% Array Structure: (No. of Segments) x (No. of Channels) x (Segment Length)
data_structure = load('Data/Data_Fold_1.mat');
train_sm = data_structure.SM_Train;
train_ocp_labels = data_structure.Y_Train;
train_sm_power_3P = train_sm(:,1,:);
% train_sm_power_1P_Combined = mean(cat(2,train_sm(:,2,:),train_sm(:,3,:),train_sm(:,4,:)),2);
% train_sm_current_1P_Combined = mean(cat(2,train_sm(:,5,:),train_sm(:,6,:),train_sm(:,7,:)),2);
run("Set_Configs.m");
warning('off')
clc;
configs = load('Config.mat');
opts = configs.opts;
feature_matrix = Extract_Features(train_sm_power_3P,opts);
feature_list = transpose(readcell('Feature_List.csv','Delimiter',','));
% Create Channel-Wise Feature Matrix and Corresponding Feature List
num_channel = size(feature_matrix,2);
feature_matrix_raveled = reshape(feature_matrix,[size(feature_matrix,1),size(feature_matrix,2)*size(feature_matrix,3)]);
feature_list_raveled = cell(1,size(feature_list,2)*num_channel);
counter = 0;
for i=1:size(feature_list,2)
    feature_list_temp = cell2mat(feature_list(1,i));
    for ii=1:num_channel
        counter = counter + 1;
        feature_list_temp_ch = strcat(feature_list_temp,'_ch_',string(ii));
        feature_list_raveled(1,counter) = {feature_list_temp_ch};
    end
end
feature_matrix = feature_matrix_raveled;
% feature_list = feature_list_raveled;
% Save Raw Feature Matrix
% save_path_mat = 'Statistical_Features_Train_Fold1.mat';
% save(save_path_mat,'feature_matrix','feature_list','opts','-v7.3');
% Combine Classic ML, DL and datetime Features
% Load Classical ML Features and Labels
% data_structure_ML = load('Statistical_Features_Train_Fold1.mat');
feature_matrix_ML = feature_matrix;
feature_list_ML = table2cell(readtable('Feature_List.csv'))';
% Load DL Features and Labels
data_structure_DL_DT_Labels = readtable('feature_list_train_Fold1.xlsx');
feature_matrix_DL_DT_Labels = table2array(data_structure_DL_DT_Labels(:,2:end));
feature_list_DL_DT_Labels = data_structure_DL_DT_Labels.Properties.VariableNames(2:end);
% Combine ML and DL Features and corresponding labels
feature_matrix = cat(2,feature_matrix_ML,feature_matrix_DL_DT_Labels);
feature_list = [feature_list_ML,feature_list_DL_DT_Labels];
% Save Feature Matrix
feature_table = array2table(feature_matrix,'VariableNames',string(feature_list));
filename = 'Features_Train_Fold1.xlsx';
writetable(feature_table,filename,'Sheet',1)
%% Test Feature Extraction, Concatenation and Saving
warning('off')
% Load Data for Feature Extraction
clc;
clear;
% Array Structure: (No. of Segments) x (No. of Channels) x (Segment Length)
data_structure = load('Data/Data_Fold_1.mat');
test_sm = data_structure.SM_Test;
test_ocp_labels = data_structure.Y_Test;
test_sm_power_3P = test_sm(:,1,:);
% test_sm_power_1P_Combined = mean(cat(2,test_sm(:,2,:),test_sm(:,3,:),test_sm(:,4,:)),2);
% test_sm_current_1P_Combined = mean(cat(2,test_sm(:,5,:),test_sm(:,6,:),test_sm(:,7,:)),2);
run("Set_Configs.m");
configs = load('Config.mat');
opts = configs.opts;
feature_matrix = Extract_Features(test_sm_power_3P,opts);
feature_list = transpose(readcell('Feature_List.csv','Delimiter',','));
% Create Channel-Wise Feature Matrix and Corresponding Feature List
num_channel = size(feature_matrix,2);
feature_matrix_raveled = reshape(feature_matrix,[size(feature_matrix,1),size(feature_matrix,2)*size(feature_matrix,3)]);
feature_list_raveled = cell(1,size(feature_list,2)*num_channel);
counter = 0;
for i=1:size(feature_list,2)
    feature_list_temp = cell2mat(feature_list(1,i));
    for ii=1:num_channel
        counter = counter + 1;
        feature_list_temp_ch = strcat(feature_list_temp,'_ch_',string(ii));
        feature_list_raveled(1,counter) = {feature_list_temp_ch};
    end
end
feature_matrix = feature_matrix_raveled;
% feature_list = feature_list_raveled;
% Save Raw Feature Matrix
% save_path_mat = 'Statistical_Features_Test_Fold1.mat';
% save(save_path_mat,'feature_matrix','feature_list','opts','-v7.3');
% Combine Classic ML, DL and datetime Features
% Load Classical ML Features and Labels
% data_structure_ML = load('Statistical_Features_Test_Fold1.mat');
feature_matrix_ML = feature_matrix;
feature_list_ML = table2cell(readtable('Feature_List.csv'))';
% Load DL Features and Labels
data_structure_DL_DT_Labels = readtable('feature_list_test_Fold1.xlsx');
feature_matrix_DL_DT_Labels = table2array(data_structure_DL_DT_Labels(:,2:end));
feature_list_DL_DT_Labels = data_structure_DL_DT_Labels.Properties.VariableNames(2:end);
% Combine ML and DL Features and corresponding labels
feature_matrix = cat(2,feature_matrix_ML,feature_matrix_DL_DT_Labels);
feature_list = [feature_list_ML,feature_list_DL_DT_Labels];
% Save Feature Matrix
feature_table = array2table(feature_matrix,'VariableNames',string(feature_list));
filename = 'Features_Test_Fold1.xlsx';
writetable(feature_table,filename,'Sheet',1)
%% Feature Correlation (Optional)
clear;
clc;
% Configurations
threshold = 0.9;
% Load Raw Features - Train
data_structure = readtable('Features_Train_Fold1.xlsx');
feature_matrix = table2array(data_structure(:,2:end));
feature_list = data_structure.Properties.VariableNames(2:end);
[feature_matrix_HC,feature_list_HC] = mycorrelm(feature_matrix,feature_list,threshold);
% Save Train Feature Matrix after Highly Correlated Feature Elimination
feature_table_HC = array2table(feature_matrix_HC,'VariableNames',string(feature_list_HC));
filename = 'Features_Train_Fold1_HCE_90.xlsx';
writetable(feature_table_HC,filename,'Sheet',1)
% Load Raw Features - Test
data_structure = readtable('Features_Test_Fold1.xlsx');
feature_matrix = table2array(data_structure(:,2:end));
feature_list = data_structure.Properties.VariableNames(2:end);
% Save Test Feature Matrix after Highly Correlated Feature Elimination
feature_table = array2table(feature_matrix,'VariableNames',string(feature_list));
feature_table_HC = feature_table(:,feature_list_HC);  % Update Feature Table
filename = 'Features_Test_Fold1_HCE_90.xlsx';
writetable(feature_table_HC,filename,'Sheet',1)
%% Feature Ranking
clear;
clc;
data_structure = load('Feature_Matrix/EEG_Seizure_HighCorr_Eliminated_Features.mat');
feature_matrix = data_structure.feature_matrix;
feature_list = data_structure.feature_list;
labels = data_structure.labels;
subject_labels = data_structure.subject_labels;
opts = data_structure.opts;
%
listFS = { ...
    'relieff','mrmr','chisquare','fscnca', ...
    'mcfs','UDFS','cfs','lasso'...
    'llcfs', 'fsasl','ufsol' ...
};
fsa_prompt =['Please, select a feature selection ', 'method from the list:'];
[methodID,~] = listdlg('PromptString',fsa_prompt,'SelectionMode','single', ...
                    'ListString',listFS,'Name','Feature Selection Method', ...
                    'ListSize', [400, 200]);
selectionMethod = listFS{methodID};
[scores, ranking] = rankfeatures(feature_matrix,labels,selectionMethod);
rankedFeatureNames = feature_list(ranking);
rankedFeatures = feature_matrix(:,ranking);
% Select Top 'n' features
n = 224;
feature_matrix = rankedFeatures(:,1:n);
feature_list = rankedFeatureNames(1,1:n);
save_path_mat = 'Feature_Matrix/EEG_Seizure_Ranked_Features.mat';
save(save_path_mat,'feature_matrix','feature_list','labels','subject_labels','opts','-v7.3');
%% Create 2D Feature Maps (Optional, for 2D Novel Pipeline) (Optional)
clear;
clc;
data_structure = load('Feature_Matrix/BCI_SAES_Ranked_Features.mat');
feature_matrix = data_structure.feature_matrix;
feature_list = data_structure.feature_list;
labels = data_structure.labels;
subject_labels = data_structure.subject_labels;
opts = data_structure.opts;
% Normalize Features for 2D
feature_matrix_norm = normalize(feature_matrix, 1, 'range', [0 1]);
% Create 2D Feature Matrix and Save as RGB Images
outdir = 'Data_2D';
if ~exist(outdir, 'dir')
    mkdir(outdir)
end
feature_size = size(feature_matrix,2);
feature_mat_2D_1 = zeros(feature_size,feature_size);
feature_mat_2D_2 = zeros(feature_size,feature_size);
for i=1:size(feature_matrix,1)
    feature_matrix_temp = feature_matrix_norm(i,:);
    label_temp = labels(i,1);
    subject_label_temp = subject_labels(i,1);
    for ii=1:size(feature_matrix,2)
        feature_mat_2D_1(ii,:) = feature_matrix_temp;
        feature_mat_2D_2(ii,:) = flip(feature_matrix_temp,2);
    end
    %
    feature_mat_2D_RGB = cat(3,feature_mat_2D_1,transpose(feature_mat_2D_1),feature_mat_2D_2);
    IMG_filename = sprintf('SAES_BCI_%d_%d.png',subject_label_temp,i);
    img_outdir = fullfile(outdir,string(label_temp));
    if ~exist(img_outdir, 'dir')
        mkdir(img_outdir)
    end
    savedir = fullfile(img_outdir,IMG_filename);
    imwrite(feature_mat_2D_RGB,savedir,'BitDepth',8,'Mode','lossless');
end