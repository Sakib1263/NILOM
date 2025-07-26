function opts = jEntropyGriddedDistribution(Sig, opts) 
% GridEn  estimates the gridded distribution entropy of a univariate data sequence.
%   
%   [GDE, GDR] = GridEn(Sig) 
%   
%   Returns the gridded distribution entropy (``GDE``) and the gridded 
%   distribution rate (``GDR``) estimated from the data sequence (``Sig``) using 
%   the default  parameters:
%   grid coarse-grain = 3, time delay = 1, logarithm = natural
%   
%   [GDE, GDR, PIx, GIx, SIx, AIx] = GridEn(Sig)
%   
%   In addition to ``GDE`` and ``GDR``, ``GridEn`` returns the following indices 
%   estimated from the data sequence (``Sig``) using the default  parameters:
%     -  ``PIx``   - Percentage of points below the line of identity (LI)
%     -  ``GIx``   - Proportion of point distances above the LI
%     -  ``SIx``   - Ratio of phase angles (w.r.t. LI) of the points above the LI
%     -  ``AIx``   - Ratio of the cumulative area of sectors of points above the LI
%   
%   [GDE, GDR, ...,] = GridEn(Sig, name, value, ...)
%   
%   Returns the gridded distribution entropy (``GDE``) estimate of the data 
%   sequence (``Sig``) using the specified name/value pair arguments:
% 
%      * ``m``     - Grid coarse-grain (``m`` x ``m`` sectors), an integer > 1
%      * ``tau``   - Time Delay, a positive integer
%      * ``Logx``  - Logarithm base, a positive scalar
%   
Sig = squeeze(Sig);
if size(Sig,1) > size(Sig,2)
    Sig = Sig';
end

if isfield(opts,'GdEn_m'), m = opts.GdEn_m; end
if isfield(opts,'GdEn_tau'), tau = opts.GdEn_tau; end
if isfield(opts,'GdEn_Logx'), Logx = opts.GdEn_Logx; end

if Logx == 0
    Logx = exp(1);
end
Sig_n = (Sig-min(Sig))/range(Sig);
Temp = [Sig_n(1:end-tau);Sig_n(1+tau:end)];
N = hist3(Temp',[m,m]);
Pj = flipud(N')/size(Temp,2); Ppi = Pj(Pj>0);
if round(sum(Ppi)) ~= 1
    warning('Potential error of estimated probabilities: P = %d', sum(Ppi))
end
GDE = -sum(Ppi.*(log(Ppi)/log(Logx)));
GDR = sum(N(:)~=0)/(m*m);
T2   = atand(Temp(2,:)./Temp(1,:))';
Dup  = sum(abs(diff(Temp(:,T2>45),[],1)));
Dtot = sum(abs(diff(Temp(:,T2~=45),[],1)));
Sup  = sum((T2(T2>45)-45));
Stot = sum(abs(T2(T2~=45)-45));
Aup  = sum(abs(((T2(T2>45)-45)).*sqrt(sum(Temp(:,T2>45).^2)')));
Atot = sum(abs(((T2(T2~=45)-45)).*sqrt(sum(Temp(:,T2~=45).^2)'))); 
PIx = 100*sum(T2 < 45)/sum(T2~=45);
GIx = 100*Dup/Dtot;
SIx = 100*Sup/Stot;
AIx = 100*Aup/Atot;
opts = cat(2,GDE,GDR,PIx,GIx,SIx,AIx);
end
