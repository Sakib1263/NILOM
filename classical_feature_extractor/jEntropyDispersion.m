function outp = jEntropyDispersion(Sig, opts)
% DispEn  estimates the dispersion entropy of a univariate data sequence.
%
%   [Dispx, RDE] = DispEn(Sig)
% 
%   Returns the dispersion entropy (``Dispx``) and the reverse dispersion entropy
%   (``RDE``) estimated from the data sequence (``Sig``) using the default parameters:
%   embedding dimension = 2, time delay = 1, symbols = 3, logarithm = natural,
%   data transform = normalised cumulative density function (ncdf)
%
%   [Dispx, RDE] = DispEn(Sig, name, value, ...)
% 
%   Returns the dispersion entropy (``Dispx``) and the reverse dispersion entropy
%   (``RDE``) estimated from the data sequence (``Sig``) using the specified
%   name/value pair arguments:
% 
%      * ``m``     - Embedding Dimension, a positive integer
%      * ``tau``   - Time Delay, a positive integer
%      * ``c``     - Number of symbols, an integer > 1
%      * ``Typex`` - Type of data-to-symbolic sequence transform, one of the following:
%        {``linear``, ``kmeans``, ``ncdf``, ``finesort``, ``equal``}
%        See the `EntropyHub Guide` for more info on these transforms.
%      * ``Logx``  - Logarithm base, a positive scalar
%      * ``Fluct`` - When ``Fluct == true``, ``DispEn`` returns the fluctuation-based
%        Dispersion entropy.   [default: false]
%      * ``Norm``  - Normalisation of ``Dispx`` and ``RDE`` values, a boolean:
%                - [false]   no normalisation - default
%                - [true]    normalises w.r.t number of possible dispersion 
%                  patterns (``c^m``  or ``(2c -1)^m-1`` if ``Fluct == true``).
%      * ``rho``   - *If ``Typex == 'finesort'``, ``rho`` is the tuning parameter (default: 1)
Sig = squeeze(Sig);
if size(Sig,1) == 1
    Sig = Sig';
end

if isfield(opts,'DpEn_m'), m = opts.DpEn_m; end
if isfield(opts,'DpEn_tau'), tau = opts.DpEn_tau; end
if isfield(opts,'DpEn_c'), c = opts.DpEn_c; end
if isfield(opts,'DpEn_Typex'), Typex = opts.DpEn_Typex; end
if isfield(opts,'DpEn_Logx'), Logx = opts.DpEn_Logx; end
if isfield(opts,'DpEn_Fluct'), Fluct = opts.DpEn_Fluct; end
if isfield(opts,'DpEn_Norm'), Norm = opts.DpEn_Norm; end
if isfield(opts,'DpEn_rho'), rho = opts.DpEn_rho; end

if Logx == 0
    Logx = exp(1);
end

N = length(Sig);
switch lower(Typex)
    case 'linear'
        Zi = discretize(Sig,linspace(min(Sig),max(Sig),c+1));
        
    case 'kmeans'    
        if size(Sig,2)>size(Sig,1)
            Sig = Sig';
        end        
        
        [Zx,Clux] = kmeans(Sig, c, 'MaxIter', 200);
        [~,xx] = sort(Clux);        Zi = zeros(1,N);
        for k = 1:c
            Zi(Zx==xx(k)) = k;
        end
        clear Clux Zx xx
        
    case 'ncdf'       
        Zx = normcdf(Sig,mean(Sig),std(Sig,1));
        Zi = discretize(Zx,linspace(0,1,c+1));     
        
    case 'finesort'
        Zx = normcdf(Sig,mean(Sig),std(Sig,1));
        Zi = discretize(Zx,linspace(0,1,c+1));
        Ym = zeros(N-(m-1)*tau, m);
        for n = 1:m
            Ym(:,n) = Zx(1+(n-1)*tau:N-((m-n)*tau));
        end
        Yi = floor(max(abs(diff(Ym,[],2)),[],2)/(rho*std(abs(diff(Sig)),1)));
        clear Zx Ym
        
    case 'equal'
        [~,ix] = sort(Sig);
        xx = round(linspace(0,N,c+1));
        Zi = zeros(1,N);
        for k = 1:c
            Zi(ix(xx(k)+1:xx(k+1))) = k;
        end
        clear ix xx
end

Zm = zeros(N-(m-1)*tau, m);
for n = 1:m
    Zm(:,n) = Zi(1+(n-1)*tau:N-((m-n)*tau));
end

if strcmpi(Typex,'finesort')
    Zm = [Zm Yi];
end
if Fluct
    Zm = diff(Zm,[],2);
    if m < 2
       warning(['Fluctuation-based Dispersion Entropy'...
           ' is undefined for m = 1. '...
           'An embedding dimension (m) > 1 should be used.'])       
    end
end

T = unique(Zm,'rows');
Nx = size(T,1);
Counter = zeros(1,Nx);
for n = 1:Nx
    Counter(n) = sum(~any(Zm - T(n,:),2));
end
Ppi = Counter(Counter~= 0)/length(Zm);
% RDE = sum(Ppi.^2) - (1/Nx);

if Fluct
    RDE = sum((Ppi - (1/((2*c - 1)^(m-1)))).^2);
else
    RDE = sum((Ppi - (1/(c^m))).^2);
end

if round(sum(Ppi),4) ~= 1
    warning('Potential Error calculating probabilities')
end

Dispx = -sum(Ppi.*log(Ppi)/log(Logx));
if Norm
    %Dispx = Dispx/(log(Nx)/log(Logx));
    if Fluct
        Dispx = Dispx/(log((2*c - 1)^(m-1))/log(Logx));
        RDE = RDE/(1 - (1/((2*c - 1)^(m-1))));
    else
        Dispx = Dispx/(log(c^m)/log(Logx));
        RDE = RDE/(1 - (1/(c^m)));
    end
end
outp = cat(2,Dispx,RDE);
end
