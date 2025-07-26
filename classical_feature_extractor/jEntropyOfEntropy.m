function opts = jEntropyOfEntropy(Sig, opts) 
% EnofEn  estimates the entropy of entropy from a univariate data sequence.
%
%   [EoE, AvEn, S2] = EnofEn(Sig) 
% 
%   Returns the entropy of entropy (``EoE``), the average Shannon entropy 
%   (``AvEn``), and the number of levels (``S2``) across all windows 
%   estimated from the data sequence (``Sig``) using the default parameters: 
%   window length (samples) = 10, slices = 10, logarithm = natural
%
%   [EoE, AvEn, S2] = EnofEn(Sig, name, value, ...)
% 
%   Returns the entropy of entropy (``EoE``) estimated from the data sequence (``Sig``)  
%   using the specified name/value pair arguments:
% 
%      * ``tau``    - Window length, an integer > 1
%      * ``S``      - Number of slices, an integer > 1
%      * ``Logx``   - Logarithm base, a positive scalar  
%
Sig = squeeze(Sig);
if size(Sig,1) > 1
    Sig = Sig';
end

if isfield(opts,'EnEn_tau'), tau = opts.EnEn_tau; end
if isfield(opts,'EnEn_S'), S = opts.EnEn_S; end
if isfield(opts,'EnEn_Logx'), Logx = opts.EnEn_Logx; end

if Logx == 0
    Logx = exp(1);
end
minSig = min(Sig,[],"all");
maxSig = max(Sig,[],"all");
Xrange = [minSig maxSig];

Wn = floor(length(Sig)/tau);
Wj = reshape(Sig(1:Wn*tau),tau,Wn)';
Yj = zeros(1,Wn);
Edges = linspace(Xrange(1),Xrange(2),S+1); % Edges = linspace(min(Sig),max(Sig),S(1)+1);

for n = 1:Wn
    Temp = histcounts(Wj(n,:),Edges)/tau;
    Temp(Temp==0) = [];
    Yj(n) = -sum(Temp.*(log(Temp)/log(Logx)));
end

AvEn = sum(Yj)/Wn;
% Edges = linspace(min(Yj),max(Yj),S+1);
% Pjl = histcounts(Yj,Edges)/Wn;
% Pjl(Pjl==0) = [];
[~,~,Tempy] = unique(round(Yj,12));
Pjl = accumarray(Tempy,1)/Wn;
if round(sum(Pjl),5) ~= 1
    warning('Possible error estimating probabilities')
end
S2 = length(Pjl);
EoE = -sum(Pjl.*(log(Pjl)/log(Logx)));
opts = EoE;
end
