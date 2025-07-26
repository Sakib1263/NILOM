function outp = jEntropyPhase(Sig, opts)
% PhasEn  estimates the phase entropy of a univariate data sequence.
%
%   [Phas] = PhasEn(Sig) 
% 
%   Returns the phase entropy (``Phas``) estimate of the data sequence (``Sig``)
%   using the default parameters: 
%   angular partitions = 4, time delay = 1, logarithm = natural,
%   normalisation = true
%
%   [Phas] = PhasEn(Sig, name, value, ...)
% 
%   Returns the phase entropy (``Phas``) estimate of the data sequence (``Sig``)  
%   using the specified name/value pair arguments:
%       * ``K``     - Angular partitions (coarse graining), an integer > 1
%                * Note: Division of partitions begins along the positive
%                        x-axis. As this point is somewhat arbitrary, it is
%                        recommended to use even-numbered (preferably
%                        multiples of 4) partitions for sake of symmetry. 
%       * ``tau``   - Time Delay, a positive integer
%       * ``Logx``  - Logarithm base, a positive scalar  
%       * ``Norm``  - Normalisation of Phas value, a boolean:
%                * [false] no normalisation
%                * [true]  normalises w.r.t. the # partitions (``Log(K)``) (Default)
%

Sig = squeeze(Sig);
if size(Sig,1) > 1
    Sig = Sig';
end

if isfield(opts,'PhEn_K'), K = opts.PhEn_K; end
if isfield(opts,'PhEn_tau'), tau = opts.PhEn_tau; end
if isfield(opts,'PhEn_Logx'), Logx = opts.PhEn_Logx; end
if isfield(opts,'PhEn_Norm'), Norm = opts.PhEn_Norm; end

if Logx == 0
    Logx = exp(1);
end

if size(Sig,1) < size(Sig,2)
    Sig = Sig';
end
Yn = Sig(1+2*tau:end) - Sig(tau+1:end-tau);
Xn = Sig(tau+1:end-tau) - Sig(1:end-2*tau);
Theta_r = atan(Yn./Xn);
Theta_r(Yn<0 & Xn<0) = Theta_r(Yn<0 & Xn<0) + pi;
Theta_r(Yn<0 & Xn>0) = Theta_r(Yn<0 & Xn>0) + 2*pi;
Theta_r(Yn>0 & Xn<0) = Theta_r(Yn>0 & Xn<0) + pi;

Angs = linspace(0,2*pi,K+1);
Tx = zeros(K,length(Theta_r));
Si = zeros(1,K);
for n = 1:K
    Temp = (Theta_r > Angs(n) & Theta_r < Angs(n+1));
    Tx(n,Temp) = 1;
    Si(n) = sum(Theta_r(Temp));
end

Si(Si==0) = [];
Phas = -sum((Si/sum(Si)).*(log(Si/sum(Si))/log(Logx)));
if Norm
    Phas = Phas/(log(K)/log(Logx));
end
outp = Phas;
end
