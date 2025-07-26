function MYOP = jMyopulsePercentageRate(X,opts)
% Parameter
thres = 0.016;    % threshold

if isfield(opts,'mthres'), thres = opts.mthres; end

N = length(X); 
Y = 0; 
for i = 1:N
  if abs(X(i)) >= thres
    Y = Y + 1;
  end
end
MYOP = Y / N;
end
