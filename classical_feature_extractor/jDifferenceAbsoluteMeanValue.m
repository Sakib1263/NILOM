function DAMV = jDifferenceAbsoluteMeanValue(X,~)
N = length(X); 
Y = 0;
for i = 1 : N - 1
  Y = Y + abs(X(i+1) - X(i));
end
DAMV = Y / (N - 1);
end
