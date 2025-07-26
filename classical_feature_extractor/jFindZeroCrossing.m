function count = jFindZeroCrossing(signal,opts)
eps = 0.5;
if isfield(opts,'eps'), eps = opts.eps; end
sum = 0;
for i=1:(size(signal,1)-1)
    if (abs(signal(i)-signal(i+1)) > eps) && ((signal(i)*signal(i+1)) < 0)
       sum = sum+1;
    end
end
count = sum;
end
