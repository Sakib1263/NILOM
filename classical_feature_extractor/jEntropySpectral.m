function outp = jEntropySpectral(Sig, opts) 
% SpecEn  estimates the spectral entropy of a univariate data sequence.
%
%   [Spec, BandEn] = SpecEn(Sig) 
% 
%   Returns the spectral entropy estimate of the full spectrum (``Spec``)
%   and the within-band entropy (``BandEn``) estimated from the data sequence 
%   (``Sig``) using the default parameters: 
%   N-point FFT = ``length(Sig)*2 + 1``, normalised band edge frequencies = [0 1],
%   logarithm = natural, normalisation = w.r.t # of spectrum/band values.
%
%   [Spec, BandEn] = SpecEn(Sig, name, value, ...)
% 
%   Returns the spectral entropy (``Spec``) and the within-band entropy (``BandEn``)
%   estimate for the data sequence (``Sig``) using the specified name/value pair arguments:
%       * ``N``'     - Resolution of spectrum (N-point FFT), an integer > 1
%       * ``Freqs`` - Normalised spectrum band edge-frequencies, a 2 element vector
%         with values in range [0 1] where 1 corresponds to the Nyquist frequency (Fs/2).
%         Note: When no band frequencies are entered, ``BandEn == SpecEn``
%       * ``Logx``  - Logarithm base, a positive scalar (default: natural log) 
%       * ``Norm``  - Normalisation of ``Spec`` value, a boolean:
%               -  [false]  no normalisation.
%               -  [true]   normalises w.r.t # of spectrum/band frequency values (default).
%
Sig = squeeze(Sig);
if size(Sig,1) == 1
    Sig = Sig';
end
if isfield(opts,'SpecEn_N'), N = opts.SpecEn_N; end
if isfield(opts,'SpecEn_Freqs'), Freqs = opts.SpecEn_Freqs; end
if isfield(opts,'SpecEn_Logx'), Logx = opts.SpecEn_Logx; end
if isfield(opts,'SpecEn_Norm'), Norm = opts.SpecEn_Norm; end

if Logx == 0
    Logx = exp(1);
end
if size(Sig,1) > size(Sig,2)
    Sig = Sig';
end
Fx = ceil(N/2);
Freqs = round(Freqs*Fx);
Freqs(Freqs==0) = 1;

if Freqs(1) > Freqs(2)
    error('Lower band frequency must come first.')
elseif diff(Freqs) < 1
    error('Spectrum resoution too low to determine bandwidth.') 
elseif min(Freqs)<0 || max(Freqs)>Fx
    error('Freqs must be normalized w.r.t sampling frequency [0 1].')
% elseif round(Freqs*ceil(N/2)) > N || round(Freqs*ceil(N/2)) < 1
%     error('Spectrum resoution too low - rounding error.')      
end
Pt = abs(fft(conv(Sig,Sig),N));
Pxx = Pt(1:Fx)/sum(Pt(1:Fx));
Spec = -(Pxx*log(Pxx)')/log(Logx);
Pband = (Pxx(Freqs(1):Freqs(2)))/sum(Pxx(Freqs(1):Freqs(2)));
BandEn = -(Pband*log(Pband)')/log(Logx);
if Norm
    Spec = Spec/(log(Fx)/log(Logx));
    BandEn = BandEn/(log(diff(Freqs)+1)/log(Logx));
end
outp = cat(2,Spec,BandEn);
end
