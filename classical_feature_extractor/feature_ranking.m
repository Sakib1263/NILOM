%% ... Constants

% clear; clc;
PATH_ROOT = "";
PATH_DATASET = 'D:\Datasets\NDD\files\gaitndd\1.0.0\';
PATH_SEGMENTS = 'D:\Datasets\NDD\files\gaitndd\1.0.0\Segments\';
PATH_FEATURES = 'D:\Datasets\NDD\files\gaitndd\1.0.0\Features\';
PATH_FIGURES = '../../Figures/';
CORR_THRESHOLD = 0.9;

%% ... Load features, labels and config
% Here the variables features, subject_ids, labels,
% featureConfig are loaded which were saved by feature_extraction.m

path = fullfile(PATH_ROOT, PATH_FEATURES);
load(fullfile(path, 'features.mat'));

% load extra features ( extracted from python etc... )
% load(fullfile(path, 'extra_features.mat'));

% allFeatures = [features, extraFeatures];
allFeatures = features;

%% Standardization

featureArray = allFeatures.Variables;
featureArray = (featureArray - mean(featureArray)) ./ std(featureArray);
allFeatures.Variables = featureArray;

allFeatures = allFeatures(:, ~any(ismissing(allFeatures))); 

%% ... Corellated feature elemation

addpath('../lib/');
featuresOptimized = mycorrelm(allFeatures, CORR_THRESHOLD);

%% ... Problem Config
% ... Converting to a N class problem, N = 2,3,4 ...

% mask = (labels == 0) | (labels == 3);
% featuresOptimized = featuresOptimized(mask, :);
% subject_ids = subject_ids(mask, :);
% labels = labels(mask, :);
% 
% labels(labels == 3) = 1;

%% Feature ranking algorithms

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

%% Perform feature ranking

addpath('../lib/feature_ranking/');

[scores, ranking] = rankfeatures( ...
    featuresOptimized.Variables, ...
    labels, ...
    selectionMethod ...
);

rankedFeatureNames = featuresOptimized.Properties.VariableNames(ranking);
rankedFeatures = featuresOptimized(:, rankedFeatureNames);

%% Plot feature ranking (Optional)

fig = plotfeatureimportance( ...
    scores, ...
    ranking, ...
    rankedFeatureNames, ...
    selectionMethod ...
);

figurePath = fullfile(PATH_ROOT, PATH_FIGURES, strcat(selectionMethod, '.png'));
saveas(fig, figurePath);

%% Save ranked features

path = fullfile(PATH_ROOT, PATH_FEATURES, 'ranked_features.mat');
save(path, 'rankedFeatures', 'featureConfig', 'selectionMethod', ...
    'subject_ids', 'labels');



