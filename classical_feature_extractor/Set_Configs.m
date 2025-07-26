%% Set Cofigurations for Feature Extraction
% General Parameters
opts.fs = 60;  % Sampling Frequency
opts.cutoff_low = 1;
opts.cutoff_high = ceil(opts.fs/2);
opts.alpha = 2;
opts.L = 2;
opts.num_int = 2;
opts.thres = 0.01;
opts.myothres = 0.2;
opts.vorder = 2;
opts.mthres = 0.016;
opts.eps = 1;
opts.lowerbound = 1;
opts.upperbound = 64;
opts.bins = 20;
% PSD Measure: Autoregressive Covariance
opts.order_autocov = 4;
% PSD Measure: Burgs
opts.order_burgs = 12;
% PSD Measure: Pyulear
opts.order_pyulear = 12;
% PSD Measure: Pspectrum
opts.cutoff_low_pspec = 1;
opts.cutoff_high_pspec  = ceil(opts.fs/2);
% Approximate Entropy (ApEn)
opts.ApEn_dim = 4;
opts.ApEn_r = 0.2;
% Attention Entropy (AtEn)
opts.AtEn_Logx = 2;
% Bubble Entropy (BbEn)
opts.BbEn_m = 2;
opts.BbEn_tau = 1;
opts.BbEn_Logx = exp(1);
% Conditional Entropy (CdEn)
opts.CdEn_m = 2;
opts.CdEn_tau = 1;
opts.CdEn_c = 6;
opts.CdEn_Logx = exp(1);  % 'e': use exp(1) or 0
opts.CdEn_Norm = 0;
% Cosine Similarity Entropy (CsEn)
opts.CsEn_m = 2;
opts.CsEn_tau = 1;
opts.CsEn_r = .1;
opts.CsEn_Logx = 2;
opts.CsEn_Norm = 3;
% Cross Approximate Entropy (XApEn)
opts.XApEn_m = 2;
opts.XApEn_tau = 1;
opts.XApEn_r = 0;
opts.XApEn_Logx = exp(1);
% Cross Conditional Entropy (XCdEn)
opts.XCdEn_m = 2;
opts.XCdEn_tau = 1;
opts.XCdEn_c = 6;
opts.XCdEn_Logx = exp(1);
opts.XCdEn_Norm = 0;
% Cross Distribution Entropy (XDsEn)
opts.XDsEn_m = 2;
opts.XDsEn_tau = 1;
opts.XDsEn_Bins = 'sturges';
opts.XDsEn_Logx = exp(1);
opts.XDsEn_Norm = 2;
% Cross Fuzzy Entropy (XFzEn)
opts.XFzEn_m = 2;
opts.XFzEn_tau = 1;
opts.XFzEn_Fx = 'default';
opts.XFzEn_r = [.2 2];
opts.XFzEn_Logx = exp(1);
% Cross Kolmogorov Entropy (XK2En)
opts.XK2En_m = 2;
opts.XK2En_tau = 1;
opts.XK2En_r = 0;
opts.XK2En_Logx = exp(1);
% Cross Permutation Entropy (XPmEn)
opts.XPmEn_m = 3;
opts.XPmEn_tau = 1;
opts.XPmEn_Logx = 2;
% Cross Sample Entropy (XSmEn)
opts.XSmEn_m = 2;
opts.XSmEn_tau = 1;
opts.XSmEn_r = 0;
opts.XSmEn_Logx = exp(1);
% Cross Spectral Entropy (XSpEn)
opts.XSpEn_N = 5;
opts.XSpEn_Freqs = [0 1];
opts.XSpEn_Logx = exp(1);
opts.XSpEn_Norm = 1;
% Dispersion Entropy (DpEn)
opts.DpEn_m = 2;
opts.DpEn_tau = 1;
opts.DpEn_c = 3;
opts.DpEn_Typex = 'ncdf';
opts.DpEn_Logx = exp(1);
opts.DpEn_Fluct = 0;
opts.DpEn_Norm = 0;
opts.DpEn_rho = 1;
% Distribution Entropy (DsEn)
opts.DsEn_m = 2;
opts.DsEn_tau = 1;
opts.DsEn_Bins = 'sturges';
opts.DsEn_Logx = 0;
opts.DsEn_Norm = 1;
% Entropy of Entropy (EnEn)
opts.EnEn_tau = 10;
opts.EnEn_S = 10;
opts.EnEn_Logx = exp(1);
% Fuzzy Entropy (ApFz)
opts.FzEn_dim = 4;
opts.FzEn_r = 0.2;
opts.FzEn_n = 2;
% Gridded Distribution Entropy (GdEn)
opts.GdEn_m = 3;
opts.GdEn_tau = 1;
opts.GdEn_Logx = exp(1);
% Increment Entropy (IcEn)
opts.IcEn_m = 2;
opts.IcEn_tau = 1;
opts.IcEn_R = 4;
opts.IcEn_Logx = 2;
opts.IcEn_Norm = 1;
% Kolmogorov Entropy (K2En)
opts.K2En_m = 2;
opts.K2En_tau = 1;
opts.K2En_r = 0;
opts.K2En_Logx = exp(1);
% Permutation Entropy (PmEn)
opts.PmEn_dim = 4;
% Phase Entropy (PhEn)
opts.PhEn_K = 4;
opts.PhEn_tau = 1;
opts.PhEn_Logx = exp(1);
opts.PhEn_Norm = 1;
% Sample Entropy (SpEn)
opts.SpEn_dim = 4;
opts.SpEn_r = 0.2;
% Slope Entropy (SlEn)
opts.SlEn_m = 2;
opts.SlEn_tau = 1;
opts.SlEn_Lvls = [5 45];
opts.SlEn_Logx = 2;
opts.SlEn_Norm = 1;
% Spectral Entropy (DsEn)
opts.SpecEn_N = 1025;
opts.SpecEn_Freqs = [0 1];
opts.SpecEn_Logx = exp(1);
opts.SpecEn_Norm = 1;
% Symbolic Dynamic Entropy (SyDyEn)
opts.SyDyEn_m = 2;
opts.SyDyEn_tau = 1;
opts.SyDyEn_c = 3;
opts.SyDyEn_Typex = 'MEP';
opts.SyDyEn_Logx = exp(1);
opts.SyDyEn_Norm = 1;
% Envelope Features (Envf)
opts.Envf_np = 300;
% Save
save('Config.mat','opts')