function outp = jSpectralStatisticalFeatures(Sig,opts)
Sig = squeeze(Sig);
if size(Sig,1) > 1
    Sig = Sig';
end
% Load Configurations
if isfield(opts,'fs'), fs = opts.fs; end
if isfield(opts,'order_autocov'), order_autocov = opts.order_autocov; end
if isfield(opts,'order_burgs'), order_burgs = opts.order_burgs; end
if isfield(opts,'order_pyulear'), order_pyulear = opts.order_pyulear; end
if isfield(opts,'cutoff_low_pspec'), cutoff_low = opts.cutoff_low_pspec; end
if isfield(opts,'cutoff_high_pspec'), cutoff_high = opts.cutoff_high_pspec; end

% PSD Measures
% Autoregressive Covariance (AutoCov)
[p_autocov,f_autocov] = pcov(Sig,order_autocov);
f_autocov = f_autocov*((fs/2)/pi);
f_autocov = f_autocov(f_autocov <= ceil(fs/2));
p_autocov = p_autocov(f_autocov <= ceil(fs/2));
ft_autocov_1 = skewness(p_autocov,0);
ft_autocov_2 = kurtosis(p_autocov,0);
ft_autocov_3 = meanfreq(p_autocov,f_autocov);
ft_autocov_4 = medfreq(p_autocov,f_autocov);
ft_autocov_5 = spectralCrest(p_autocov,f_autocov);
ft_autocov_6 = spectralDecrease(p_autocov,f_autocov);
ft_autocov_7 = spectralFlatness(p_autocov,f_autocov);
ft_autocov_8 = max(spectralFlux(Sig',fs));
ft_autocov_9 = spectralKurtosis(p_autocov,f_autocov);
ft_autocov_10 = spectralRolloffPoint(p_autocov,f_autocov);
ft_autocov_11 = spectralSkewness(p_autocov,f_autocov);
ft_autocov_12 = spectralSlope(p_autocov,f_autocov);
ft_autocov_13 = spectralSpread(p_autocov,f_autocov);
ft_autocov_14 = spectralCentroid(p_autocov,f_autocov);
outp_autocov = [ft_autocov_1 ft_autocov_2 ft_autocov_3 ft_autocov_4 ft_autocov_5 ft_autocov_6...
    ft_autocov_7 ft_autocov_8 ft_autocov_9 ft_autocov_10 ft_autocov_11 ft_autocov_12... 
    ft_autocov_13 ft_autocov_14];
% Burgs
[p_burgs,f_burgs] = pburg(Sig,order_burgs);
f_burgs = f_burgs*((fs/2)/pi);
f_burgs = f_burgs(f_burgs <= ceil(fs/2));
p_burgs = p_burgs(f_burgs <= ceil(fs/2));
ft_burgs_1 = skewness(p_burgs,0);
ft_burgs_2 = kurtosis(p_burgs,0);
ft_burgs_3 = meanfreq(p_burgs,f_burgs);
ft_burgs_4 = medfreq(p_burgs,f_burgs);
ft_burgs_5 = spectralCrest(p_burgs,f_burgs);
ft_burgs_6 = spectralDecrease(p_burgs,f_burgs);
ft_burgs_7 = spectralFlatness(p_burgs,f_burgs);
ft_burgs_8 = max(spectralFlux(Sig',fs));
ft_burgs_9 = spectralKurtosis(p_burgs,f_burgs);
ft_burgs_10 = spectralRolloffPoint(p_burgs,f_burgs);
ft_burgs_11 = spectralSkewness(p_burgs,f_burgs);
ft_burgs_12 = spectralSlope(p_burgs,f_burgs);
ft_burgs_13 = spectralSpread(p_burgs,f_burgs);
ft_burgs_14 = spectralCentroid(p_burgs,f_burgs);
outp_burgs = [ft_burgs_1 ft_burgs_2 ft_burgs_3 ft_burgs_4 ft_burgs_5 ft_burgs_6...
    ft_burgs_7 ft_burgs_8 ft_burgs_9 ft_burgs_10 ft_burgs_11 ft_burgs_12... 
    ft_burgs_13 ft_burgs_14];
% Lombscale
[p_lombscale,f_lombscale] = plomb(Sig,fs,'power');
f_lombscale = f_lombscale(f_lombscale <= ceil(fs/2));
p_lombscale = p_lombscale(f_lombscale <= ceil(fs/2));
ft_lombscale_1 = skewness(p_lombscale,0);
ft_lombscale_2 = kurtosis(p_lombscale,0);
ft_lombscale_3 = meanfreq(p_lombscale,f_lombscale);
ft_lombscale_4 = medfreq(p_lombscale,f_lombscale);
ft_lombscale_5 = spectralCrest(p_lombscale,f_lombscale);
ft_lombscale_6 = spectralDecrease(p_lombscale,f_lombscale);
ft_lombscale_7 = spectralFlatness(p_lombscale,f_lombscale);
ft_lombscale_8 = max(spectralFlux(Sig',fs));
ft_lombscale_9 = spectralKurtosis(p_lombscale,f_lombscale);
ft_lombscale_10 = spectralRolloffPoint(p_lombscale,f_lombscale);
ft_lombscale_11 = spectralSkewness(p_lombscale,f_lombscale);
ft_lombscale_12 = spectralSlope(p_lombscale,f_lombscale);
ft_lombscale_13 = spectralSpread(p_lombscale,f_lombscale);
ft_lombscale_14 = spectralCentroid(p_lombscale,f_lombscale);
outp_lombscale = [ft_lombscale_1 ft_lombscale_2 ft_lombscale_3 ft_lombscale_4 ft_lombscale_5 ft_lombscale_6...
    ft_lombscale_7 ft_lombscale_8 ft_lombscale_9 ft_lombscale_10 ft_lombscale_11 ft_lombscale_12... 
    ft_lombscale_13 ft_lombscale_14];
% Multitaper
[p_multitaper,f_multitaper] = pmtm(Sig,fs);
f_multitaper = f_multitaper*((fs/2)/pi);
f_multitaper = f_multitaper(f_multitaper <= ceil(fs/2));
p_multitaper = p_multitaper(f_multitaper <= ceil(fs/2));
ft_multitaper_1 = skewness(p_multitaper,0);
ft_multitaper_2 = kurtosis(p_multitaper,0);
ft_multitaper_3 = meanfreq(p_multitaper,f_multitaper);
ft_multitaper_4 = medfreq(p_multitaper,f_multitaper);
ft_multitaper_5 = spectralCrest(p_multitaper,f_multitaper);
ft_multitaper_6 = spectralDecrease(p_multitaper,f_multitaper);
ft_multitaper_7 = spectralFlatness(p_multitaper,f_multitaper);
ft_multitaper_8 = max(spectralFlux(Sig',fs));
ft_multitaper_9 = spectralKurtosis(p_multitaper,f_multitaper);
ft_multitaper_10 = spectralRolloffPoint(p_multitaper,f_multitaper);
ft_multitaper_11 = spectralSkewness(p_multitaper,f_multitaper);
ft_multitaper_12 = spectralSlope(p_multitaper,f_multitaper);
ft_multitaper_13 = spectralSpread(p_multitaper,f_multitaper);
ft_multitaper_14 = spectralCentroid(p_multitaper,f_multitaper);
outp_multitaper = [ft_multitaper_1 ft_multitaper_2 ft_multitaper_3 ft_multitaper_4 ft_multitaper_5 ft_multitaper_6...
    ft_multitaper_7 ft_multitaper_8 ft_multitaper_9 ft_multitaper_10 ft_multitaper_11 ft_multitaper_12... 
    ft_multitaper_13 ft_multitaper_14];
% Periodogram
[p_periodogram,f_periodogram] = periodogram(Sig);
f_periodogram = f_periodogram*((fs/2)/pi);
f_periodogram = f_periodogram(f_periodogram <= ceil(fs/2));
p_periodogram = p_periodogram(f_periodogram <= ceil(fs/2));
ft_periodogram_1 = skewness(p_periodogram,0);
ft_periodogram_2 = kurtosis(p_periodogram,0);
ft_periodogram_3 = meanfreq(p_periodogram,f_periodogram);
ft_periodogram_4 = medfreq(p_periodogram,f_periodogram);
ft_periodogram_5 = spectralCrest(p_periodogram,f_periodogram);
ft_periodogram_6 = spectralDecrease(p_periodogram,f_periodogram);
ft_periodogram_7 = spectralFlatness(p_periodogram,f_periodogram);
ft_periodogram_8 = max(spectralFlux(Sig',fs));
ft_periodogram_9 = spectralKurtosis(p_periodogram,f_periodogram);
ft_periodogram_10 = spectralRolloffPoint(p_periodogram,f_periodogram);
ft_periodogram_11 = spectralSkewness(p_periodogram,f_periodogram);
ft_periodogram_12 = spectralSlope(p_periodogram,f_periodogram);
ft_periodogram_13 = spectralSpread(p_periodogram,f_periodogram);
ft_periodogram_14 = spectralCentroid(p_periodogram,f_periodogram);
outp_periodogram = [ft_periodogram_1 ft_periodogram_2 ft_periodogram_3 ft_periodogram_4 ft_periodogram_5 ft_periodogram_6...
    ft_periodogram_7 ft_periodogram_8 ft_periodogram_9 ft_periodogram_10 ft_periodogram_11 ft_periodogram_12... 
    ft_periodogram_13 ft_periodogram_14];
% Pspectrum
[p_pspectrum,f_pspectrum] = pspectrum(Sig,fs,'FrequencyLimits',[cutoff_low cutoff_high]);
f_pspectrum = f_pspectrum(f_pspectrum <= ceil(fs/2));
p_pspectrum = p_pspectrum(f_pspectrum <= ceil(fs/2));
ft_pspectrum_1 = skewness(p_pspectrum,0);
ft_pspectrum_2 = kurtosis(p_pspectrum,0);
ft_pspectrum_3 = meanfreq(p_pspectrum,f_pspectrum);
ft_pspectrum_4 = medfreq(p_pspectrum,f_pspectrum);
ft_pspectrum_5 = spectralCrest(p_pspectrum,f_pspectrum);
ft_pspectrum_6 = spectralDecrease(p_pspectrum,f_pspectrum);
ft_pspectrum_7 = spectralFlatness(p_pspectrum,f_pspectrum);
ft_pspectrum_8 = max(spectralFlux(Sig',fs));
ft_pspectrum_9 = spectralKurtosis(p_pspectrum,f_pspectrum);
ft_pspectrum_10 = spectralRolloffPoint(p_pspectrum,f_pspectrum);
ft_pspectrum_11 = spectralSkewness(p_pspectrum,f_pspectrum);
ft_pspectrum_12 = spectralSlope(p_pspectrum,f_pspectrum);
ft_pspectrum_13 = spectralSpread(p_pspectrum,f_pspectrum);
ft_pspectrum_14 = spectralCentroid(p_pspectrum,f_pspectrum);
outp_pspectrum = [ft_pspectrum_1 ft_pspectrum_2 ft_pspectrum_3 ft_pspectrum_4 ft_pspectrum_5 ft_pspectrum_6...
    ft_pspectrum_7 ft_pspectrum_8 ft_pspectrum_9 ft_pspectrum_10 ft_pspectrum_11 ft_pspectrum_12... 
    ft_pspectrum_13 ft_pspectrum_14];
% Pyulear
[p_pyl,f_pyl] = pyulear(Sig,order_pyulear);
f_pyl = f_pyl*((fs/2)/pi);
f_pyl = f_pyl(f_pyl <= ceil(fs/2));
p_pyl = p_pyl(f_pyl <= ceil(fs/2));
ft_pyl_1 = skewness(p_pyl,0);
ft_pyl_2 = kurtosis(p_pyl,0);
ft_pyl_3 = meanfreq(p_pyl,f_pyl);
ft_pyl_4 = medfreq(p_pyl,f_pyl);
ft_pyl_5 = spectralCrest(p_pyl,f_pyl);
ft_pyl_6 = spectralDecrease(p_pyl,f_pyl);
ft_pyl_7 = spectralFlatness(p_pyl,f_pyl);
ft_pyl_8 = max(spectralFlux(Sig',fs));
ft_pyl_9 = spectralKurtosis(p_pyl,f_pyl);
ft_pyl_10 = spectralRolloffPoint(p_pyl,f_pyl);
ft_pyl_11 = spectralSkewness(p_pyl,f_pyl);
ft_pyl_12 = spectralSlope(p_pyl,f_pyl);
ft_pyl_13 = spectralSpread(p_pyl,f_pyl);
ft_pyl_14 = spectralCentroid(p_pyl,f_pyl);
outp_pyl = [ft_pyl_1 ft_pyl_2 ft_pyl_3 ft_pyl_4 ft_pyl_5 ft_pyl_6...
    ft_pyl_7 ft_pyl_8 ft_pyl_9 ft_pyl_10 ft_pyl_11 ft_pyl_12... 
    ft_pyl_13 ft_pyl_14];
% Welch
[p_welch,f_welch] = pwelch(Sig,fs,'power');
f_welch = f_welch*((fs/2)/pi);
f_welch = f_welch(f_welch <= ceil(fs/2));
p_welch = p_welch(f_welch <= ceil(fs/2));
ft_welch_1 = skewness(p_welch,0);
ft_welch_2 = kurtosis(p_welch,0);
ft_welch_3 = meanfreq(p_welch,f_welch);
ft_welch_4 = medfreq(p_welch,f_welch);
ft_welch_5 = spectralCrest(p_welch,f_welch);
ft_welch_6 = spectralDecrease(p_welch,f_welch);
ft_welch_7 = spectralFlatness(p_welch,f_welch);
ft_welch_8 = max(spectralFlux(Sig',fs));
ft_welch_9 = spectralKurtosis(p_welch,f_welch);
ft_welch_10 = spectralRolloffPoint(p_welch,f_welch);
ft_welch_11 = spectralSkewness(p_welch,f_welch);
ft_welch_12 = spectralSlope(p_welch,f_welch);
ft_welch_13 = spectralSpread(p_welch,f_welch);
ft_welch_14 = spectralCentroid(p_welch,f_welch);
outp_welch = [ft_welch_1 ft_welch_2 ft_welch_3 ft_welch_4 ft_welch_5 ft_welch_6...
    ft_welch_7 ft_welch_8 ft_welch_9 ft_welch_10 ft_welch_11 ft_welch_12... 
    ft_welch_13 ft_welch_14];
%
outp = [outp_autocov outp_burgs outp_lombscale outp_multitaper... 
    outp_periodogram outp_pspectrum outp_pyl outp_welch];
end
