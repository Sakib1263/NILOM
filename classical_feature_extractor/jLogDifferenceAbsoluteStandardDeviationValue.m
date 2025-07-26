function LDASDV = jLogDifferenceAbsoluteStandardDeviationValue(X,~)
N = length(X); 
Y = 0;
for t = 1 : N - 1
  Y = Y + (X(t+1) - X(t)) ^ 2;
end
LDASDV = log(sqrt(Y / (N - 1)));
end
