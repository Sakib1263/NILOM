function TsEn = jEntropyTsallis(X,opts)
% Parameter
alpha = 2;     % alpha

if isfield(opts,'alpha'), alpha = opts.alpha; end

% Convert probability using energy 
C    = (X .^ 2) ./ sum(X .^ 2);
% Entropy 
En   = C .^ alpha;
TsEn = (1 / (alpha - 1)) * (1 - sum(En(:))); 
end
