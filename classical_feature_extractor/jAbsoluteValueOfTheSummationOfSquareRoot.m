function ASS = jAbsoluteValueOfTheSummationOfSquareRoot(X,~)
temp = sum(X .^ (1/2));
ASS  = abs(temp);
end
