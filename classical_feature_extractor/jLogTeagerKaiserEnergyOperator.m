function LTKEO = jLogTeagerKaiserEnergyOperator(X,~)
N = length(X); 
Y = 0; 
for j = 2 : N - 1
  Y = Y + ((X(j) ^ 2) - X(j-1) * X(j+1));
end
LTKEO = abs(log(Y));
end
