function outp = jDominantFrequencyFeatures(data, opts)
% This function estimates the dominant frequency of each channel (COLUMN) of the
% input data.  First, it calculates the power spectrum of the signal.  Then it finds the 
% maximum frequency over as well as the "energy" at that point; options below allow normalization
% by the total or mean energy.  Note this does not restrict the range of frequencies to search 
% for a maximum.
% 
% data         matrix containing data where each COLUMN corresponds to a channel
% fs           sampling rate (Hz)
% cutoff       cutoff frequency (Hz)
% maxfreq      frequency at which the max of the spectrum occurs (Hz) (ROW VECTOR)
% maxratio     ratio of the energy of the maximum to the total energy (ROW VECTOR)
% Copyright (c) 2016, MathWorks, Inc.

if isfield(opts,'fs'), fs = opts.fs; end
if isfield(opts,'cutoff_high'), cutoff = opts.cutoff_high; end

[~, nchannels] = size(data);

nfft = 2^nextpow2(length(data));
% nfft = 1024;
f = fs/2*linspace(0,1,nfft/2);
cutoffidx = length(find(f <= cutoff));
f = f(1:cutoffidx);

% calculate the power spectrum using FFT method
datafft = fft(data,nfft);
ps = real(datafft.*conj(datafft));

% keep only the non-redundant portion
ps = ps(1:cutoffidx,:);
for ich = 1:nchannels
    ps(:,ich) = ps(:,ich)/sum(ps(:,ich));
end

% locate max value below cutoff
[maxval, maxind] = max(ps);
maxfreq = f(maxind);

% calculate peak energy by summing energy from maxfreq-delta to maxfreq+delta
% then normalize by total energy below cutoff
delta = 5;  % Hz
maxratio = zeros(1,nchannels);
for ich = 1:nchannels
    maxrange = f>=maxfreq(ich)-delta & f<=maxfreq(ich)+delta;
    maxratio(ich) = sum(ps(maxrange,ich)) / sum(ps(:,ich));
end
outp = cat(1,maxval,maxratio);
end