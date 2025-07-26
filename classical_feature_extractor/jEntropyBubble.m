function outp = jEntropyBubble(Sig, opts)
% BubbEn  estimates the bubble entropy of a univariate data sequence.
%
%   [Bubb, H] = BubbEn(Sig)
% 
%   Returns the bubble entropy (``Bubb``) and the conditional Renyi entropy (``H``)
%   estimates from the data sequence (``Sig``) using the default parameters:
%   embedding dimension = 2, time delay = 1, logarithm = natural
%
%   [Bubb, H] = BubbEn(Sig, name, value, ...)
% 
%   Returns the bubble entropy (``Bubb``) estimated from the data sequence (``Sig``)
%   using the specified name/value pair arguments:
% 
%       * ``m``     - Embedding Dimension, an integer > 1.  
%         ``BubbEn`` returns estimates for each dimension [2, ..., ``m``]
% 
%       * ``tau``   - Time Delay, a positive integer
%       * ``Logx``  - Logarithm base, a positive scalar
%
Sig = squeeze(Sig);
if size(Sig,1) > 1
    Sig = Sig';
end

if isfield(opts,'BbEn_m'), m = opts.BbEn_m; end
if isfield(opts,'BbEn_tau'), tau = opts.BbEn_tau; end
if isfield(opts,'BbEn_Logx'), Logx = opts.BbEn_Logx; end

if Logx == 0
    Logx = exp(1);
end

N = length(Sig);
Sx = zeros(N,m+1);
H = zeros(1,m+1);
Sx(:,1) = Sig;
for k = 2:m+1
    Sx(1:N-(k-1)*tau,k) = Sig(1+(k-1)*tau:N);
    [Swapx] = BubbSort(Sx(1:N-(k-1)*tau,1:k));
    [~,~,Locs] = unique(Swapx);
    p = accumarray(Locs,1)/(N-(k-1)*tau);
    H(k) = -log(sum(p.^2))/log(Logx);
    
    if round(sum(p)) ~= 1
        warning('Potential error in detected swap number')
    end
    clear Swapx p Locs
end

Bubb = diff(H)./(log((2:m+1)./(0:m-1))/log(Logx));
Bubb(1) = [];
outp = Bubb;
end

function [swaps, bsorted] = BubbSort(Data)

[x,N2] = size(Data);
swaps = zeros(1,x);
for y = 1:x
    t = 1;
    while t <= N2-1
        for kk = 1:N2-t
            if Data(y,kk) > Data(y,kk+1)
                temp = Data(y,kk);
                Data(y,kk) = Data(y,kk+1);
                Data(y,kk+1) = temp;
                swaps(y) = swaps(y) + 1;
            end
        end
        t = t + 1;
    end
end
bsorted = Data;
end
