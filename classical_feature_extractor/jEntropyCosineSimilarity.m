function outp = jEntropyCosineSimilarity(Sig, opts)
% CoSiEn  estimates the cosine similarity entropy of a univariate data sequence.
%
%   [CoSi, Bm] = CoSiEn(Sig) 
% 
%   Returns the cosine similarity entropy (``CoSi``) and the corresponding
%   global probabilities (``Bm``) estimated from the data sequence (``Sig``) 
%   using the default parameters: 
%   embedding dimension = 2, time delay = 1, angular threshold = .1,
%   logarithm = base 2,
%
%   [CoSi, Bm] = CoSiEn(Sig, name, value, ...)
% 
%   Returns the cosine similarity entropy (``CoSi``) estimated from the data
%   sequence (``Sig``) using the specified name/value pair arguments:
% 
%       * ``m``     - Embedding Dimension, an integer > 1
%       * ``tau``   - Time Delay, a positive integer
%       * ``r``     - Angular threshold, a value in range [0 < ``r`` < 1]
%       * ``Logx``  - Logarithm base, a positive scalar (enter 0 for natural log) 
%       * ``Norm``  - Normalisation of ``Sig``, one of the following integers:
%               *  [0]  no normalisation - default
%               *  [1]  remove median(``Sig``) to get zero-median series
%               *  [2]  remove mean(``Sig``) to get zero-mean series
%               *  [3]  normalises ``Sig`` w.r.t. SD(``Sig``)
%               *  [4]  normalises ``Sig`` values to range [-1 1]
%

Sig = squeeze(Sig);
if size(Sig,1) == 1
    Sig = Sig';
end

if isfield(opts,'CsEn_m'), m = opts.CsEn_m; end
if isfield(opts,'CsEn_tau'), tau = opts.CsEn_tau; end
if isfield(opts,'CsEn_r'), r = opts.CsEn_r; end
if isfield(opts,'CsEn_Logx'), Logx = opts.CsEn_Logx; end
if isfield(opts,'CsEn_Norm'), Norm = opts.CsEn_Norm; end

if Logx == 0
    Logx = exp(1);
end
N = length(Sig);
if Norm == 1
    Xi = Sig - median(Sig);
elseif Norm == 2
    Xi = Sig - mean(Sig);
elseif Norm == 3
    Xi = (Sig - mean(Sig))/std(Sig,1);
elseif Norm == 4
    Xi = (2*(Sig - min(Sig))/range(Sig)) - 1;
else
    Xi = Sig;
end
Nx = N-((m-1)*tau);
Zm = zeros(Nx,m);
for n = 1:m
    Zm(:,n) = Xi((n-1)*tau+1:Nx+(n-1)*tau);
end

Num = Zm*Zm'; 
Mag = sqrt(diag(Num));
Den = Mag*Mag';
AngDis = acos(Num./Den)/pi;
if max(imag(AngDis(:))) < (10^-5) % max(max(imag(AngDis))) < (10^-5)
    Bm = sum(sum(triu(round(AngDis,6) < r,1)))/(Nx*(Nx-1)/2);
else
    Bm = sum(sum(triu(real(AngDis) < r,1)))/(Nx*(Nx-1)/2);
    warning('Complex values ignored')
end
if Bm == 1 || Bm == 0
    CoSi = 0;
else
    CoSi = -(Bm*log(Bm)/log(Logx)) - ((1-Bm)*log(1-Bm)/log(Logx));
end
outp = CoSi;
end
