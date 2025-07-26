function [y,baseline_fit,baseline] = Fix_Baseline_Drift_2(x,order)
    sig_length = length(x);  % Find the Signal Length for Time Vector Creation
    sig_size = size(x);
    if sig_size(1) > 1
        x = x';
    end
    time = linspace(1,sig_length,sig_length)';
    %
    baseline = movmin(x,20);          % Get Lower Bounds
    p = polyfit(time',baseline,order);  % Polynomial Fit of the Baseline provides with better performance
    baseline_fit = polyval(p,time'); % Polynomial Fit of the Baseline
    y_temp = x - baseline_fit;       % Subtract the baseline from the original signal
    y = y_temp - min(y_temp);
end