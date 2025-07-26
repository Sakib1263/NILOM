function [ApEn] = jEntropyApproximate(series,opts)
% Checking the ipunt parameters:
control = ~isempty(series);
assert(control,'The user must introduce a time series (first inpunt).');
control = ~isempty(opts.ApEn_dim);
assert(control,'The user must introduce a embbeding dimension (second inpunt).');
control = ~isempty(opts.ApEn_r);
assert(control,'The user must introduce a tolerand (r) (third inpunt).');
if isfield(opts,'ApEn_dim'), dim = opts.ApEn_dim; end
if isfield(opts,'ApEn_r'), r = opts.ApEn_r; end
% Processing:
N = length(series);
result = zeros(1,2);
r = r*std(series);
for j = 1:2
    m = dim+j-1; % 'm' is the embbeding dimension used each iteration
    % Pre-definition of the varialbes for computational efficiency:
    phi = zeros(1,N-m+1);
    patterns = zeros(m,N-m+1);
    % First, we compose the patterns
    % The columns of the matrix 'patterns' will be the (N-m+1) patterns of 'm' length:
    if m == 1 % If the embedding dimension is 1, each sample is a pattern:
        patterns = series;
    else % Otherwise, we build the patterns of length 'm':
        for i = 1:m
            patterns(i,:) = series(i:N-m+i);
        end
    end
    % Second, we compute the number of patterns whose distance is less than the tolerance.
    % This loop goes over the columns of matrix 'patterns':
    for i = 1:N-m+1
        % 'temp' is an auxiliar matrix whose elements are the maximum 
        % absolut difference between the current pattern and the rest:
        if m == 1 
            temp = abs(patterns - repmat(patterns(:,i),1,N-m+1));
        else
            temp = max(abs(patterns - repmat(patterns(:,i),1,N-m+1)));
        end
        % We determine which elements of 'temp' are smaller than the tolerance:
        bool = any((temp < r),1);
        % We get the relative frequency of the current pattern: 
        phi(i) = sum(bool)/(N-m+1);
    end
    % Finally, we average the natural logarithm of all relative frequencies:
    result(j) = mean(log(phi));
end
ApEn = result(1)-result(2);
end
