function outp = jDPratioPspectrum(Sig,opts)
Sig = squeeze(Sig);
if size(Sig,1) > 1
    Sig = Sig';
end

if isfield(opts,'fs'), fs = opts.fs; end

cutoff_low = 1;
cutoff_high = ceil(fs/2);
[p,f] = pspectrum(Sig,fs,'FrequencyLimits',[cutoff_low cutoff_high]);
f = f(f <= ceil(fs/2));
p = p(f <= ceil(fs/2));
f_all = f(f>=1 & f<=ceil(fs/2));
p_all = p(f>=1 & f<=64);

% Average PSD over N points using N/2 points before and after
N = 13;
b = ones(N,1)/N;
a = 1;

% All
mean_psd = filter(b,a,[p_all;zeros(floor(N/2),1)]);
mean_psd = mean_psd(floor(N/2) + (1:length(f_all)));
highest_mean_psd = max(mean_psd,[],"all");
lowest_mean_psd = min(mean_psd,[],"all");
DP_all = 10*log10(highest_mean_psd/lowest_mean_psd);

outp = DP_all;
end