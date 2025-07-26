function TM = jTemporalMoment(X,opts)
% Parameter
order = 3;    % order

if isfield(opts,'order'), order = opts.order; end

N  = length(X);
TM = abs((1 / N) * sum(X .^ order));
end
