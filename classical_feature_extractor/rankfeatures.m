function [weights, ranking] = rankfeatures(train_x, train_y, selection_method)
% problems in 'dgufs',ILFS,'rfe' ,'L0','fisher','ECFS','InfFS'
tic
% number of features
numF = size(train_x,2);
% change 'tol' in 'fitrgp' to modify fsa
% change 'k' in 'relieff' to modify fsa

fprintf('\n---------------Performing Feature Selection---------------\n')
% feature Selection on training data
switch lower(selection_method)
    case 'inffs'
        % Infinite Feature Selection 2015 updated 2016
        alpha = 0.5;    % default, it should be cross-validated.
        sup = 1;        % Supervised or Not
        [ranking, weights] = infFS( train_x , train_y, alpha , sup , 0 );
        
    case 'ilfs'
        % Infinite Latent Feature Selection - ICCV 2017
        [ranking, weights] = ILFS(train_x, train_y , 6, 0 );
        
    case 'fsasl'
        options.lambda1 = 1;
        options.LassoType = 'SLEP';
        options.SLEPrFlag = 1;
        options.SLEPreg = 0.01;
        options.LARSk = 5;
        options.LARSratio = 2;
        nClass=2;
        [W, ~] = FSASL(train_x', nClass, options);
        [weights,ranking]=sort(abs(W(:,1))+abs(W(:,2)),'descend');
        
        % ... fix weights order
        idx = 1:length(ranking);
        temp(ranking) = idx;
        weights = weights(temp);
        
    case 'lasso'
        lambda = 25;
        B = lasso(train_x,train_y);
        [weights,ranking]=sort(B(:,lambda),'descend');
        
        % ... fix weights order
        idx = 1:length(ranking);
        temp(ranking) = idx;
        weights = weights(temp);
        
    case 'ufsol'
        para.p0 = 'sample';
        para.p1 = 1e6;
        para.p2 = 1e2;
        nClass = 2;
        [~,~,ranking,~,weights] = UFSwithOL(train_x',nClass,para) ;
        
        % ... fix weights order
        idx = 1:length(ranking);
        temp(ranking) = idx;
        weights = weights(temp);
        
    case 'dgufs'
        S = dist(train_x');
        S = -S./max(max(S)); % it's a similarity
        nClass = 2;
        alpha = 0.5;
        beta = 0.9;
        nSel = 2;
        [Y,~] = DGUFS(train_x',nClass,S,alpha,beta,nSel);
        [weights,ranking]=sort(Y(:,1)+Y(:,2),'descend');
                
    case 'mrmr'
        [ranking, weights] = fscmrmr(train_x, train_y);
        
    case 'relieff'
        k = 20;
        [ranking, weights] = relieff( train_x, train_y, k);
        
    case 'mutinffs'
        [ ranking , weights] = mutInfFS( train_x, train_y, numF );
        
    case 'fsv'
        [ ranking , weights] = fsvFS( train_x, train_y, numF );
        
    case 'laplacian'
        W = dist(train_x');
        W = -W./max(max(W)); % it's a similarity
        [lscores] = LaplacianScore(train_x, W);
        %[weights, ranking] = sort(-lscores);
        [weights, ranking] = sort(lscores, 'descend'); % changed to be compatible 
                                                       % with other algo
    case 'mcfs'
        % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
        options = [];
        options.k = 5; %For unsupervised feature selection, you should tune
        %this parameter k, the default k is 5.
        options.nUseEigenfunction = 4;  %You should tune this parameter.
        [FeaIndex,~, weights] = MCFS_p(train_x,numF,options);
        ranking = FeaIndex{1};
        
        % ... fix weights order
        idx = 1:length(ranking);
        temp(ranking) = idx;
        weights = weights(temp);
        
    case 'rfe'
        ranking = spider_wrapper(train_x,train_y,numF,lower(selection_method));
        
    case 'l0'
        ranking = spider_wrapper(train_x,train_y,numF,lower(selection_method));
        
    case 'fisher'
        ranking = spider_wrapper(train_x,train_y,numF,lower(selection_method));
              
    case 'ecfs'
        % Features Selection via Eigenvector Centrality 2016
        alpha = 0.5; % default, it should be cross-validated.
        [ranking,weights] = ECFS( train_x, train_y, alpha )  ;
        
    case 'udfs'
        % Regularized Discriminative Feature Selection for Unsupervised Learning
        nClass = 2;
        [ranking, weights] = UDFS(train_x , nClass );
        
        % ... fix weights order
        idx = 1:length(ranking);
        temp(ranking) = idx;
        weights = weights(temp);
        
    case 'cfs'
        % BASELINE - Sort features according to pairwise correlations
        [weights,ranking] = cfs(train_x);
        
        % ... fix weights order
        idx = 1:length(ranking);
        temp(ranking) = idx;
        weights = weights(temp);
        
    case 'llcfs'
        % Feature Selection and Kernel Learning for Local Learning-Based Clustering
        [weights,ranking] = llcfs( train_x );
        
        % ... fix weights order
        idx = 1:length(ranking);
        temp(ranking) = idx;
        weights = weights(temp);
        
    case 'fitrgp'
        % Feature Ranking using fitrgp
        tol = 1e-4;
        verbosity = 1;
        old_weights = selectUsingGPR(train_x,train_y,tol,verbosity);
        [weights, ranking] = sort(old_weights,'descend');
        
    case 'chisquare'
        [ranking,weights] = fscchi2(train_x,train_y);    
    case 'fscnca'
        nca = fscnca(train_x,train_y);
        weights = nca.FeatureWeights; 
        clear nca
        x = (1:size(train_x,2))'; 
        a=[weights,x];
        dummy=sortrows(a,1,'descend');
        ranking=dummy(:,2);    
    otherwise
        disp('Unknown method.')
end
time_elapsed = toc;
fprintf('Total time taken for feature selection: %.3f \n',time_elapsed);