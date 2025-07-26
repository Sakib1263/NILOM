function outp = jOHMratioMultitaper(input,opts)

if isfield(opts,'fs'), fs = opts.fs; end

[p,f] = pmtm(input,fs);
f = f*((fs/2)/pi);

f = f(f <= ceil(fs/2));
p = p(f <= ceil(fs/2));

f_all = f(f>=1 & f<=ceil(fs/2));
p_all = p(f>=1 & f<=64);
M0 = sum(p_all);
M1 = sum(p_all.*f_all);
M2 = sum(p_all.*f_all.^2);
OHM_all = 10*log10(sqrt(M2/M0)/(M1/M0));

outp = OHM_all;
end