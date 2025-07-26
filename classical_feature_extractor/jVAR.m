function VAR=jVAR(X,~)
N=length(X); 
VAR=(1/(N-1))*sum(X.^2);
end
