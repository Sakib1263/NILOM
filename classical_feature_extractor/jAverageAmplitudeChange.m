function AAC = jAverageAmplitudeChange(X,~)
N = length(X); 
Y = 0;
for i = 1 : N - 1
  Y = Y + abs(X(i + 1) - X(i));
end
AAC = Y / N;
end
