function DVARV = jDifferenceVarianceValue(X,~)
N = length(X); 
Y = 0;
for i = 1 : N - 1
  Y = Y + (X(i+1) - X(i)) ^ 2;
end
DVARV = Y / (N - 2);
end
