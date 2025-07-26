function VO = jVOrder(X,opts)
% Parameter
order = 2;     % order

if isfield(opts,'vorder'), order = opts.vorder; end

N  = length(X); 
Y  = (1 / N) * sum(X .^ order);
VO = Y ^ (1 / order); 
end
