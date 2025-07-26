function outp = jEntropySymbolicDynamic(Sig, opts) 
% SyDyEn  estimates the symbolic dynamic entropy of a univariate data sequence.
% 
%   [SyDy, Zt] = SyDyEn(Sig) 
% 
%   Returns the symbolic dynamic entropy (``SyDy``) and the symbolic sequence
%   (``Zt``) of the data sequence (``Sig``) using the default parameters: 
%   embedding dimension = 2, time delay = 1, symbols = 3, logarithm = natural,
%   symbolic partition type = maximum entropy partitioning (MEP), 
%   normalisation = normalises w.r.t # possible vector permutations (``c^m``) 
% 
%   [SyDy, Zt] = SyDyEn(Sig, name, value, ...)
% 
%   Returns the symbolic dynamic entropy (``SyDy``) and the symbolic sequence
%   (``Zt``) estimated of the data sequence (``Sig``) using the specified 
%   name/value pair arguments:
% 
%       * ``m``     - Embedding Dimension, a positive integer
%       * ``tau``   - Time Delay, a positive integer
%       * ``c``     - Number of symbols, an integer > 1
%       * ``Typex`` - Type of symbolic sequence partitioning, one of the following:
%         {``'linear'``, ``'uniform'``, ``'MEP'`` (default), ``'kmeans'``}    
%       * ``Logx``  - Logarithm base, a positive scalar  
%       * ``Norm``  - Normalisation of ``SyDyEn`` value, a boolean:
%                * [false]  no normalisation 
%                * [true]   normalises w.r.t # possible vector permutations 
%                  (``c^m+1``) - default
% 
%   See the `EntropyHub guide <https://github.com/MattWillFlood/EntropyHub/blob/main/EntropyHub%20Guide.pdf>`_
%   for more info.
% 
%   See also:
%       DispEn, PermEn, CondEn, SampEn, MSEn.

Sig = squeeze(Sig);
if size(Sig,1) > 1
    Sig = Sig';
end

if isfield(opts,'SyDyEn_m'), m = opts.SyDyEn_m; end
if isfield(opts,'SyDyEn_tau'), tau = opts.SyDyEn_tau; end
if isfield(opts,'SyDyEn_c'), c = opts.SyDyEn_c; end
if isfield(opts,'SyDyEn_Typex'), Typex = opts.SyDyEn_Typex; end
if isfield(opts,'SyDyEn_Logx'), Logx = opts.SyDyEn_Logx; end
if isfield(opts,'SyDyEn_Norm'), Norm = opts.SyDyEn_Norm; end

if Logx == 0
    Logx = exp(1);
end

N = length(Sig);
Nx = N-((m-1)*tau);
switch lower(Typex)
    case 'linear' 
        % Splits amplitudes into equal bins and finds points in each bin. 
        Zt = discretize(Sig, linspace(min(Sig),max(Sig),c+1));
        
    case 'uniform'
        % Splits the data into bins with ~equal numbers per bins
        [~, Ix] = sort(Sig);
        z = discretize(1:N,linspace(1,N,c+1));
        Zt(Ix) = z;        
        
    case 'kmeans'
        % Splits data based on k-means clusters 
        [z,xx] = kmeans(Sig', c, 'MaxIter', 200);
        [~,ix] = sort(xx);    Zt = zeros(1,N);
        for k = 1:c
            Zt(z==ix(k)) = k;
        end
        clear z k Temp ix xx
        
    otherwise % MEP method is default
        Tx = sort(Sig,'ascend');
        Zt = discretize(Sig, Tx([1,ceil((1:c-1)*N/c),N]));
end

Zm = zeros(Nx,m);
for n = 1:m
    Zm(:,n) = Zt((n-1)*tau + 1:Nx+(n-1)*tau);
end

T = unique(Zm,'rows');
Counter = zeros(1,size(T,1));
Counter2 = zeros(size(T,1),c);
Bins = 0.5:c+.5;
for n = 1:size(T,1)
    Ordx = ~any(Zm - T(n,:),2);
    Counter(n) = sum(Ordx)/Nx;
    Temp = Zm([false(m*tau,1); Ordx(1:end-(m*tau))],1);
    Counter2(n,:) = histcounts(Temp,Bins,'Normalization','probability');
end

Counter2(isnan(Counter2)) = 0;
P1 = -sum(Counter.*log(Counter)/log(Logx));
P2 = log(repmat(Counter',1,c).*Counter2)/log(Logx);
P2(isinf(P2)) = 0;
SyDy = P1 - Counter*sum(P2,2);

if round(sum(Counter),4) ~= 1 || max(round(sum(Counter2,2),4)) ~= 1
    warning('Potential Error calculating probabilities.')
end
if Norm
    SyDy = SyDy/(log(c^(m+1))/log(Logx));
end
outp = SyDy;
end
