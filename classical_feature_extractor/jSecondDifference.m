function SD = jSecondDifference(X,~)
T = length(X); 
Y = 0;
for t = 1 : T - 2
  Y = Y + abs(X(t+2) - X(t));
end
SD = (1 / (T - 2)) * Y;
end
