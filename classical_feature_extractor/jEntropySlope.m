function opts = jEntropySlope(Sig, opts) 
% SlopEn  estimates the slope entropy of a univariate data sequence.
%
%   [Slop] = SlopEn(Sig) 
% 
%   Returns the slope entropy (``Slop``) estimates for embedding dimensions
%   [2, ..., ``m``] of the data sequence (``Sig``) using the default parameters: 
%   embedding dimension = 2, time delay = 1, angular thresholds = [5 45], 
%   logarithm = base 2 
% 
%   [Slop] = SlopEn(Sig, name, value, ...)
% 
%   Returns the slope entropy (``Slop``) estimate of the data sequence (``Sig``)  
%   using the specified name/value pair arguments:
% 
%       * ``m``     - Embedding Dimension, an integer > 1. 
%         ``SlopEn`` returns estimates for each dimension [2, ..., ``m``]
%       * ``tau``   - Time Delay, a positive integer
%       * ``Lvls``  - Angular thresolds, a vector of monotonically increasing
%         values in the range [0 90] degrees.
%       * ``Logx``  - Logarithm base, a positive scalar (enter 0 for natural log)
%       * ``Norm``  - Normalisation of ``Slop`` value, a boolean:
%               *  [false]  no normalisation
%               *  [true]   normalises w.r.t. the number of patterns found (default)
% 
Sig = squeeze(Sig);
if size(Sig,1) > 1
    Sig = Sig';
end

if isfield(opts,'SlEn_m'), m = opts.SlEn_m; end
if isfield(opts,'SlEn_tau'), tau = opts.SlEn_tau; end
if isfield(opts,'SlEn_Lvls'), Lvls = opts.SlEn_Lvls; end
if isfield(opts,'SlEn_Logx'), Logx = opts.SlEn_Logx; end
if isfield(opts,'SlEn_Norm'), Norm = opts.SlEn_Norm; end

if Logx == 0
    Logx = exp(1);
end
m = m-1;
Tx = atand(Sig(1+tau:end)-Sig(1:end-tau));
N = length(Tx);
Sx = zeros(N,m);
Symbx = zeros(size(Tx));
Slop = zeros(1,m);
Lvls = sort(Lvls,'ascend');

for q = 2:length(Lvls)
    Symbx(Tx<= Lvls(q) & Tx> Lvls(q-1)) = q-1;
    Symbx(Tx>=-Lvls(q) & Tx<-Lvls(q-1)) = -(q-1);
    
    if q == length(Lvls)
        Symbx(Tx> Lvls(q)) = q;
        Symbx(Tx<-Lvls(q)) = -q;
    end
end

for k = 1:m
    Sx(1:N-k+1,k) = Symbx(k:N);
    [~,~,Locs] = unique(Sx(1:N-k+1,1:k),'rows');
    
    if Norm
        p = accumarray(Locs,1)/(N-k+1);
        if round(sum(p))~= 1
            warning('Potential Error: Some permutations not accounted for!')
        end
    else
        p = accumarray(Locs,1)/numel(accumarray(Locs,1));
    end
    
    Slop(k) = -sum(p.*(log(p)/log(Logx)));
    clear Locs p
end
opts = Slop;
end
