function FuzzyEn = jEntropyFuzzy(series,opts)
% Checking the ipunt parameters:
control = ~isempty(series);
assert(control,'The user must introduce a time series (first inpunt).');
control = ~isempty(opts.FzEn_dim);
assert(control,'The user must introduce a embbeding dimension (second inpunt).');
control = ~isempty(opts.FzEn_r);
assert(control,'The user must introduce a width for the fuzzy exponential function: r (third inpunt).');
control = ~isempty(opts.FzEn_n);
assert(control,'The user must introduce a step for the fuzzy exponential function: n (fourth inpunt).');
if isfield(opts,'FzEn_dim'), dim = opts.FzEn_dim; end
if isfield(opts,'FzEn_r'), r = opts.FzEn_r; end
if isfield(opts,'FzEn_n'), n = opts.FzEn_n; end
% Processing:
% Normalization of the input time series:
series = (series-mean(series))/std(series);
N = length(series);
phi = zeros(1,2);
% Value of 'r' in case of not normalized time series:
% r = r*std(series);
for j = 1:2
    m = dim+j-1; % 'm' is the embbeding dimension used each iteration
    % Pre-definition of the varialbes for computational efficiency:
    patterns = zeros(m,N-m+1);
    aux = zeros(1,N-m+1);
    % First, we compose the patterns
    % The columns of the matrix 'patterns' will be the (N-m+1) patterns of 'm' length:
    if m == 1 % If the embedding dimension is 1, each sample is a pattern
        patterns = series;
    else % Otherwise, we build the patterns of length 'm':
        for i = 1:m
            patterns(i,:) = series(i:N-m+i);
        end
    end
    % We substract the baseline of each pattern to itself:
    for i = 1:N-m+1
        patterns(:,i) = patterns(:,i) - (mean(patterns(:,i)));
    end

    % This loop goes over the columns of matrix 'patterns':
    for i = 1:N-m
        % Second, we compute the maximum absolut distance between the
        % scalar components of the current pattern and the rest:
        if m == 1 
            dist = abs(patterns - repmat(patterns(:,i),1,N-m+1));
        else
            dist = max(abs(patterns - repmat(patterns(:,i),1,N-m+1)));
        end
       % Third, we get the degree of similarity:
       simi = exp(((-1)*((dist).^n))/r);
       % We average all the degrees of similarity for the current pattern:
       aux(i) = (sum(simi)-1)/(N-m-1); % We substract 1 to the sum to avoid the self-comparison
    end

    % Finally, we get the 'phy' parameter as the as the mean of the first
    % 'N-m' averaged drgees of similarity:
    phi(j) = sum(aux)/(N-m);
end
FuzzyEn = log(phi(1)) - log(phi(2));
end
