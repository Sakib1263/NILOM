function MSR = jMeanValueOfTheSquareRoot(X,~)
K   = length(X); 
MSR = (1 / K) * sum(X .^ (1/2));
end
