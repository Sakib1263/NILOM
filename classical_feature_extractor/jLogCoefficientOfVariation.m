function LCOV = jLogCoefficientOfVariation(X,~)
mu   = mean(X); 
sd   = std(X);
LCOV = log(sd / mu);
end
