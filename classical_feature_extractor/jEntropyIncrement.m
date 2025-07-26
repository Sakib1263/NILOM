function outp = jEntropyIncrement(Sig, opts)
% IncrEn  estimates the increment entropy of a univariate data sequence.
%
%   [Incr] = IncrEn(Sig) 
% 
%   Returns the increment entropy (``Incr``) estimated from the data sequence 
%   (``Sig``) using the default parameters: 
%   embedding dimension = 2, time delay = 1, quantifying resolution = 4,
%   logarithm = base 2,
%
%   [Incr] = IncrEn(Sig, name, value, ...)
% 
%   Returns the increment entropy (``Incr``) estimated from the data sequence 
%   (``Sig``) using the specified name/value pair arguments:
% 
%      * ``m``     - Embedding Dimension, an integer > 1
%      * ``tau``   - Time Delay, a positive integer
%      * ``R``     - Quantifying resolution, a positive integer
%      * ``Logx``  - Logarithm base, a positive scalar (enter 0 for natural log) 
%      * ``Norm``  - Normalisation of IncrEn value, a boolean:
%                * [false]  no normalisation - default
%                * [true]   normalises w.r.t embedding dimension (m-1). 
%

Sig = squeeze(Sig);
if size(Sig,1) == 1
    Sig = Sig';
end

if isfield(opts,'IcEn_m'), m = opts.IcEn_m; end
if isfield(opts,'IcEn_tau'), tau = opts.IcEn_tau; end
if isfield(opts,'IcEn_R'), R = opts.IcEn_R; end
if isfield(opts,'IcEn_Logx'), Logx = opts.IcEn_Logx; end
if isfield(opts,'IcEn_Norm'), Norm = opts.IcEn_Norm; end

if Logx == 0
    Logx = exp(1);
end

Vi = diff(Sig);
N = length(Vi)-((m-1)*tau);
Vk = zeros(N,m);
for k = 1:m
    Vk(:,k) = Vi(1+(k-1)*tau:N+(k-1)*tau);
end

Sk = sign(Vk);
Temp = std(Vk,[],2);
Qk = min(R, floor((abs(Vk)*R)./repmat(Temp,1,m)));
Qk(any(Temp==0,2),:) = 0;
Wk = Sk.*Qk;
Px = unique(Wk,'rows');
Counter = zeros(1,size(Px,1));
for k = 1:size(Px,1) 
    Counter(k) = sum(~any(Wk - Px(k,:),2));
end
Ppi = Counter/N;

if size(Px,1) > (2*R + 1)^m
    warning('Error with probability estimation')
elseif round(sum(Ppi),3) ~= 1
    warning('Error with probability estimation')
end
Incr = -sum(Ppi.*(log(Ppi)/log(Logx)));
if Norm
    Incr = Incr/(m-1);
end
outp = Incr;
end
