function outp = jEntropyKolmogorov(Sig, opts) 
% K2En  estimates the Kolmogorov (K2) entropy of a univariate data sequence.
%
%   [K2, Ci] = K2En(Sig)
% 
%   Returns the Kolmogorov entropy estimates (``K2``) and the correlation
%   integrals (``Ci``) for ``m`` = [1,2] estimated from the data sequence (``Sig``)
%   using the default parameters: embedding dimension = 2, time delay = 1, 
%   distance threshold (``r``) = 0.2*SD(``Sig``), logarithm = natural
%
%   [K2, Ci] = K2En(Sig, name, value, ...)
% 
%   Returns the Kolmogorov entropy estimates (``K2``) for dimensions = [1, ..., ``m``]
%   estimated from the data sequence (``Sig``) using the specified name/value pair
%   arguments:
% 
%       * ``m``     - Embedding Dimension, a positive integer
%       * ``tau``   - Time Delay, a positive integer
%       * ``r``     - Radius Distance Threshold, a positive scalar  
%       * ``Logx``  - Logarithm base, a positive scalar  

Sig = squeeze(Sig);
if size(Sig,1) == 1
    Sig = Sig';
end

if isfield(opts,'K2En_m'), m = opts.K2En_m; end
if isfield(opts,'K2En_tau'), tau = opts.K2En_tau; end
if isfield(opts,'K2En_r'), r = opts.K2En_r; end
if isfield(opts,'K2En_Logx'), Logx = opts.K2En_Logx; end

if Logx == 0
    Logx = exp(1);
end

N = length(Sig);
if r == 0
    r = 0.2*std(Sig);
end
m = m+1;
Zm = zeros(N,m);
Ci = zeros(1,m);
for n = 1:m
    N2 = N-(n-1)*tau;
    Zm(1:N2,n) = Sig((n-1)*tau + 1:N);   
    Norm = zeros(N2-1);    
    for k = 1:N2-1
        Temp = repmat(Zm(k,1:n),N2-k,1) - Zm(k+1:N2,1:n);
        Norm(k,k:N2-1) = sqrt(sum(Temp.*Temp,2)); 
    end
    Norm(Norm==0) = inf;
    Ci(n) = 2*sum(sum(Norm < r))/(N2*(N2-1));     
end
 
K2 = (log(Ci(1:m-1)./Ci(2:m))/log(Logx))/tau;
K2(isinf(K2)) = NaN;
outp = K2;
end
