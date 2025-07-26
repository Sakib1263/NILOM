function outp = jEntropyAttention(Sig, opts)
% AttnEn  estimates the attention entropy of a univariate data sequence.
%
%   [Attn] = AttnEn(Sig) 
% 
%   Returns the attention entropy (``Attn``) calculated as the average of the
%   sub-entropies (``Hxx``, ``Hxn``, ``Hnn``, ``Hnx``) estimated from the data
%   sequence (``Sig``)  using a base-2 logarithm.
% 
%   [Attn, Hxx, Hnn, Hxn, Hnx] = AttnEn(Sig, 'Logx', value)
% 
%   Returns the attention entropy (``Attn``) and the sub-entropies 
%   (``Hxx``, ``Hxn``, ``Hnn``, ``Hnx``) from the data sequence (``Sig``) where,
% 
%      * ``Hxx``  -   entropy of local-maxima intervals
%      * ``Hnn``  -   entropy of local minima intervals
%      * ``Hxn``  -   entropy of intervals between local maxima and subsequent minima
%      * ``Hnx``  -   entropy of intervals between local minima and subsequent maxima
%        with the following name/value pair argument:
%      * ``Logx``  - Logarithm base, a positive scalar  (enter 0 for natural log)

Sig = squeeze(Sig);
if size(Sig,1) == 1
    Sig = Sig';
end

if isfield(opts,'AtEn_Logx'), Logx = opts.AtEn_Logx; end

Xmax = PkFind(Sig);
Xmin = PkFind(-Sig);
Txx = diff(Xmax);
Tnn = diff(Xmin);
Temp = diff(sort([Xmax; Xmin]));

if isempty(Xmax) 
   error('No local maxima found!') 
elseif isempty(Xmin)
    error('No local minima found!') 
end

if Xmax(1) < Xmin(1)
   Txn = Temp(1:2:end);
   Tnx = Temp(2:2:end);
else
   Txn = Temp(2:2:end);
   Tnx = Temp(1:2:end);
end

Pnx = histcounts(Tnx)/length(Tnx); Pnx(Pnx==0) = [];
Pnn = histcounts(Tnn)/length(Tnn); Pnn(Pnn==0) = [];
Pxx = histcounts(Txx)/length(Txx); Pxx(Pxx==0) = [];
Pxn = histcounts(Txn)/length(Txn); Pxn(Pxn==0) = [];

Hxx = -sum(Pxx.*(log(Pxx)/log(Logx)));
Hxn = -sum(Pxn.*(log(Pxn)/log(Logx)));
Hnx = -sum(Pnx.*(log(Pnx)/log(Logx)));
Hnn = -sum(Pnn.*(log(Pnn)/log(Logx)));

Attn = (Hnn + Hxx + Hxn + Hnx)/4;

outp = Attn;
end

function [Indx] = PkFind(X)
Nx = length(X);
Indx = zeros(Nx-2,1);
for n = 2:Nx-1
    if (X(n-1)< X(n)) && (X(n) > X(n+1))
        Indx(n) = n;
    elseif (X(n-1)< X(n)) && (X(n) == X(n+1))
        k = 1;
        while (n+k)<Nx && (X(n) == X(n+k))
            k = k+1;
        end
        if (X(n) > X(n+k))
            Indx(n) = n + floor((k-1)/2);
        end
    end
end

Indx(Indx==0)=[];
end
