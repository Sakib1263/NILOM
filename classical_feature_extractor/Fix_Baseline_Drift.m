function [y,baseline_fit] = Fix_Baseline_Drift(x, baseline_rate)
    sig_length = length(x);  % Find the Signal Length for Time Vector Creation
    sig_size = size(x);
    if sig_size(1) > 1
        x = x';
    end
    time = linspace(1,sig_length,sig_length)';
    %
    [p,~,mu] = polyfit(time',x,baseline_rate);  % Polynomial Fit of the Baseline provides with better performance
    baseline_fit = polyval(p,time',[],mu); % Polynomial Fit of the Baseline
    y_temp = x - baseline_fit;       % Subtract the baseline from the original signal
    y = y_temp - min(y_temp);        % Subtract any remaining DC Shift
end