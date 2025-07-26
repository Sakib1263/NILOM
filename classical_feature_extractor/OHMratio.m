% OHMratio This computes the spectral deformation
%
% OHM = OHMratio(f,p)
%
% Author Adrian Chan
%
% This computes the spectral deformation defined as:
% OHM = sqrt(M2/M0)/(M1/M0)
%
% where Mn is the nth spectral moment defined as:
% Mn = sum Pi*(fi^n)
%
% where Pi is the power spectral density at frequency fi.
%
% The spectral deformation is computed on the spectrum below 1000 Hz and is
% sensitive to symmetry and peaking in the power spectrum and to additive
% disturbances in the low and high frequency ranges.
%
% Reference: Sinderby C, Lindstrom L, Grassino AE, "Automatic assessment of
% electromyogram quality", Journal of Applied Physiology, vol. 79, no. 5,
% pp. 1803-1815, 1995.
%
% Inputs
%    f: frequencies (Hz)
%    p: power spectral density values
%
% Outputs
%    OHM: spectral deformation
%
% Modifications
% 09/09/21 AC First created.
function OHM = OHMratio(f,p)

% remove frequencies above 1000 Hz
index_below_1000 = find(f <= 1000);
f = f(index_below_1000);
p = p(index_below_1000);

M0 = sum(p);
M1 = sum(p.*f);
M2 = sum(p.*f.^2);

OHM = sqrt(M2/M0)/(M1/M0);
