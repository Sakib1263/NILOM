function outp = jEnvelopeFeatures(inp,opts)
if isfield(opts,'Envf_np'), np = opts.Envf_np; end
X = envelope(inp, np, 'peak');
% X = smoothdata(normalize(X,2),2);
f1X = max(X)/min(X);
% Calculate finalValue/min value feature
f2X = X(length(X))/min(X);
% Calculate time to peak feature in percentage of stance phas
[~,PeakIndex] = max(X);
f3X = X(PeakIndex)/length(X);
f4X = max(X);
outp = [f1X f2X f3X f4X];
end
