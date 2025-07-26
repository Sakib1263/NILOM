function LD = jLogDetector(X,~)
N = length(X); 
Y = 0;
for k = 1:N
  Y = Y + log10(abs(X(k))); 
end
LD = exp(Y / N);
end
