function LogEn = jEntropyLogEnergy(X,~)
% Entropy 
logV = log2(X.^2); 
LogEn = sum(logV(~isinf(logV)));
end
