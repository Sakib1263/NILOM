function Feat = jTDD(S,~)
[samples,channels]=size(S);

if channels>samples
    S = S';
    [samples,channels]=size(S);
end

% Root squared zero order moment normalized
m0 = sqrt(sum(S.^2));
m0 = m0.^.1/.1;

% Prepare derivatives for higher order moments
d1 = diff([zeros(1,channels);diff(S)],1,1);
d2 = diff([zeros(1,channels);diff(d1)],1,1);

% Root squared 2nd and 4th order moments normalized
m2 = sqrt(sum(d1.^2)./(samples-1));
m2 = m2.^.1/.1;

m4 = sqrt(sum(d2.^2)./(samples-1));
m4 = m4.^.1/.1;

% Sparseness
sparsi = (sqrt(abs((m0-m2).*(m0-m4))).\m0);

% Irregularity Factor
IRF = m2./sqrt(m0.*m4);

% Waveform length ratio
WLR = sum(abs(d1))./sum(abs(d2));

% All features together
Feat = log(abs(m0));
end