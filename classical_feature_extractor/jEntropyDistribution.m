function outp = jEntropyDistribution(Sig, opts)
% DistEn  estimates the distribution entropy of a univariate data sequence.
%
%   [Dist, Ppi] = DistEn(Sig) 
% 
%   Returns the distribution entropy estimate (``Dist``) and the
%   corresponding distribution probabilities (``Ppi``) estimated from the data 
%   sequence (``Sig``) using the default  parameters: 
%   embedding dimension = 2, time delay = 1, binning method = ``'Sturges'``,
%   logarithm = base 2, normalisation = w.r.t # of histogram bins
%
%   [Dist, Ppi] = DistEn(Sig, name, value, ...)
% 
%   Returns the distribution entropy estimate (``Dist``) estimated from the data
%   sequence (``Sig``) using the specified name/value pair arguments:
% 
%       * ``m``     - Embedding Dimension, a positive integer
%       * ``tau``   - Time Delay, a positive integer
%       * ``Bins``  - Histogram bin selection method for distance distribution, either
%         an integer > 1 indicating the number of bins, or one of the following strings 
%         {``'sturges'``, ``'sqrt'``, ``'rice'``, ``'doanes'``} [default: ``'sturges'``]
%       * ``Logx``  - Logarithm base, a positive scalar (enter 0 for natural log) 
%       * ``Norm``  - Normalisation of ``Dist`` value, a boolean:
%                 - [false]  no normalisation.
%                 - [true]   normalises w.r.t # of histogram bins (default)

Sig = squeeze(Sig);
if size(Sig,1) == 1
    Sig = Sig';
end

if isfield(opts,'DsEn_m'), m = opts.DsEn_m; end
if isfield(opts,'DsEn_tau'), tau = opts.DsEn_tau; end
if isfield(opts,'DsEn_Bins'), Bins = opts.DsEn_Bins; end
if isfield(opts,'DsEn_Logx'), Logx = opts.DsEn_Logx; end
if isfield(opts,'DsEn_Norm'), Norm = opts.DsEn_Norm; end

Nx = length(Sig) - ((m-1)*tau);
Zm = zeros(Nx,m);
for n = 1:m
    Zm(:,n) = Sig((n-1)*tau + 1:Nx+(n-1)*tau);
end

DistMat = zeros(1,Nx*(Nx-1)/2);
for k = 1:Nx-1
    Ix = [((k-1)*(Nx - k/2)+1), k*(Nx-((k+1)/2))];
    DistMat(Ix(1):Ix(2)) = max(abs(repmat(Zm(k,:),Nx-k,1) - Zm(k+1:end,:)),[],2);
end

Ny = length(DistMat);
if ischar(Bins)
    switch lower(Bins)
        case 'sturges'
            Bx = ceil(log2(Ny) + 1);
        case 'rice'
            Bx = ceil(2*(Ny^(1/3)));
        case 'sqrt'
            Bx = ceil(sqrt(Ny));
        case 'doanes'
            sigma = sqrt(6*(Ny-2)/((Ny+1)*(Ny+3)));
            Bx = ceil(1+log2(Ny)+log2(1+abs(skewness(DistMat)/sigma)));
        otherwise 
            error('Please enter a valid binning method')
    end
else    
    Bx = Bins;
end
By = linspace(min(DistMat),max(DistMat),Bx+1);
Ppi = histcounts(DistMat,By)/Ny;

if round(sum(Ppi),6) ~= 1
    warning('Potential error estimating probabilities (p=%d).', sum(Ppi))
    Ppi(Ppi==0)=[];
elseif any(Ppi==0)
    % fprintf('Note: %d/%d bins were empty',sum(Ppi==0),numel(Ppi));
    Ppi(Ppi==0)=[];
end
Dist = -sum(Ppi.*log(Ppi)/log(Logx));
if Norm == 1
    Dist = Dist/(log(Bx)/log(Logx));
end

outp = Dist;
end
