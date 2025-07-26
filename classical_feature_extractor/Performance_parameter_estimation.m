%% Evaluate EEG Restoration Data -- Phase 1
warning('off');
clear;
clc;
% Configs
Fs = 256;  % Sampling Frequency
% Load Data
EEG_Data_Filename = 'EEG_Restored_Phase1_UNet256_NLayerDiscriminator_Overall.h5';
EEG_GT_Segmented = transpose(squeeze(h5read(EEG_Data_Filename,'/EEG_Clean'))); % load EEG ground truth clean
EEG_Corrupted_Segmented = transpose(squeeze(h5read(EEG_Data_Filename,'/EEG_Corrupted'))); % load EEG ground truth corrupted
EEG_Pred_Segmented = transpose(squeeze(h5read(EEG_Data_Filename,'/EEG_Restored'))); % load EEG restored by CycleGAN
EEG_Pred_Corrupted_Segmented = transpose(squeeze(h5read(EEG_Data_Filename,'/EEG_Destroyed'))); % load EEG destroyed by CycleGAN
% Reshape Vectors
EEG_GT = normalize(reshape(EEG_GT_Segmented, [size(EEG_GT_Segmented,1)*size(EEG_GT_Segmented,2),1]),'range');
EEG_Corrupted = normalize(reshape(EEG_Corrupted_Segmented, [size(EEG_Corrupted_Segmented,1)*size(EEG_Corrupted_Segmented,2),1]),'range');
EEG_Pred = normalize(reshape(EEG_Pred_Segmented, [size(EEG_Pred_Segmented,1)*size(EEG_Pred_Segmented,2),1]),'range');
EEG_Pred_Corrupted = normalize(reshape(EEG_Pred_Corrupted_Segmented, [size(EEG_Pred_Corrupted_Segmented,1)*size(EEG_Pred_Corrupted_Segmented,2),1]),'range');
% Calculate Temporal Metrics
del_snr = 10*log10(var(EEG_GT)/var(EEG_Pred)) - 10*log10(var(EEG_GT)/var(EEG_Corrupted));
corr_af = corrcoef(EEG_GT,EEG_Pred);
corr_bf = corrcoef(EEG_GT,EEG_Corrupted);
corr_imprvmnt = 100 * (1 - (1 - corr_af(1,2))/(1 - corr_bf(1,2)));
% Calculate Spectral Metrics
[pxx,fx] = periodogram(EEG_GT,rectwin(size(EEG_GT,1)),size(EEG_GT,1),Fs,'psd');
[pyy,fy] = periodogram(EEG_Corrupted,rectwin(size(EEG_Corrupted,1)),size(EEG_Corrupted,1),Fs,'psd');
[pzz,fz] = periodogram(EEG_Pred,rectwin(size(EEG_Pred,1)),size(EEG_Pred,1),Fs,'psd');
Index = find(fx>1,1,'first')-1;
pxx = pxx(1:Index*120);
pyy = pyy(1:Index*120);
pzz = pzz(1:Index*120);
pxx_all = pxx(Index*1:Index*80);
pyy_all = pyy(Index*1:Index*80);
pzz_all = pzz(Index*1:Index*80);
pxx_delta = pxx(Index*1:Index*4);
pyy_delta = pyy(Index*1:Index*4);
pzz_delta = pzz(Index*1:Index*4);
pxx_theta = pxx(Index*4:Index*8);
pyy_theta = pyy(Index*4:Index*8);
pzz_theta = pzz(Index*4:Index*8);
pxx_alpha = pxx(Index*8:Index*13);
pyy_alpha = pyy(Index*8:Index*13);
pzz_alpha = pzz(Index*8:Index*13);
pxx_beta = pxx(Index*13:Index*30);
pyy_beta = pyy(Index*13:Index*30);
pzz_beta = pzz(Index*13:Index*30);
pxx_gamma = pxx(Index*30:Index*80);
pyy_gamma = pyy(Index*30:Index*80);
pzz_gamma = pzz(Index*30:Index*80);
%
delta_ratio_bf = abs(sum(pxx_delta)/sum(pxx_all));
delta_ratio_abf = abs(sum(pyy_delta)/sum(pyy_all));
delta_ratio_af = abs(sum(pzz_delta)/sum(pzz_all));
theta_ratio_bf = abs(sum(pxx_theta)/sum(pxx_all));
theta_ratio_abf = abs(sum(pyy_theta)/sum(pyy_all));
theta_ratio_af = abs(sum(pzz_theta)/sum(pzz_all));
alpha_ratio_bf = abs(sum(pxx_alpha)/sum(pxx_all));
alpha_ratio_abf = abs(sum(pyy_alpha)/sum(pyy_all));
alpha_ratio_af = abs(sum(pzz_alpha)/sum(pzz_all));
beta_ratio_bf = abs(sum(pxx_beta)/sum(pxx_all));
beta_ratio_abf = abs(sum(pyy_beta)/sum(pyy_all));
beta_ratio_af = abs(sum(pzz_beta)/sum(pzz_all));
gamma_ratio_bf = abs(sum(pxx_gamma)/sum(pxx_all));
gamma_ratio_abf = abs(sum(pyy_gamma)/sum(pyy_all));
gamma_ratio_af = abs(sum(pzz_gamma)/sum(pzz_all));
pxx = 10*log10(pxx);
pyy = 10*log10(pyy);
pzz = 10*log10(pzz);
%
corr_af_spec = corrcoef(pxx_all,pzz_all);
corr_bf_spec = corrcoef(pxx_all,pyy_all);
corr_imprvmnt_spec = 100 * (1 - (1 - corr_af_spec(1,2))/(1 - corr_bf_spec(1,2)));
MAE_Corrupted = mean(abs(EEG_GT-EEG_Corrupted),"all");
MAE_Corrupted_SD = std(abs(EEG_GT-EEG_Corrupted));
MAE_Pred = mean(abs(EEG_GT-EEG_Pred),"all");
MAE_Pred_SD = std(abs(EEG_GT-EEG_Pred));
MSE_Corrupted = mean((EEG_GT-EEG_Corrupted).^2,"all");
MSE_Corrupted_SD = std((EEG_GT-EEG_Corrupted).^2);
MSE_Pred = mean((EEG_GT-EEG_Pred).^2,"all");
MSE_Pred_SD = std((EEG_GT-EEG_Pred).^2);
RMSE_Corrupted = sqrt(mean((EEG_GT-EEG_Corrupted).^2,"all"));
RMSE_Corrupted_SD = sqrt(std((EEG_GT-EEG_Corrupted).^2));
RMSE_Pred = sqrt(mean((EEG_GT-EEG_Pred).^2,"all"));
RMSE_Pred_SD = sqrt(std((EEG_GT-EEG_Pred).^2));
% Entropy
ApEn_Clean_GT_TOT = zeros(size(EEG_GT_Segmented,1),1);
ApEn_Corrupted_TOT = zeros(size(EEG_GT_Segmented,1),1);
ApEn_Restored_TOT = zeros(size(EEG_GT_Segmented,1),1);
for i = 1:size(EEG_GT_Segmented,1)  % Loop through segments
    EEG_GT_Temp = normalize(EEG_GT_Segmented(i,:),'range');
    EEG_Corrupted_Temp = normalize(EEG_Corrupted_Segmented(i,:),'range');
    EEG_Pred_Temp = normalize(EEG_Pred_Segmented(i,:),'range');
    EEG_Pred_Corrupted_Temp = normalize(EEG_Pred_Corrupted_Segmented(i,:),'range');
    %
    ApEn_Clean_GT = ApEn(EEG_GT_Temp,5,0.1);
    ApEn_Corrupted = ApEn(EEG_Corrupted_Temp,5,0.1);
    ApEn_Restored = ApEn(EEG_Pred_Temp,5,0.1);
    ApEn_Clean_GT_TOT(i,1) = ApEn_Clean_GT;
    ApEn_Corrupted_TOT(i,1) = ApEn_Corrupted;
    ApEn_Restored_TOT(i,1) = ApEn_Restored;
end
ApEn_Clean_GT = mean(ApEn_Clean_GT_TOT,'all',"omitnan");
ApEn_Corrupted = mean(ApEn_Corrupted_TOT,'all',"omitnan");
ApEn_Restored = mean(ApEn_Restored_TOT,'all',"omitnan");
disp(ApEn_Clean_GT);
disp(ApEn_Corrupted);
disp(ApEn_Restored);
clear ApEn_Clean_GT_TOT ApEn_Corrupted_TOT ApEn_Restored_TOT;
%
SampEn_Clean_GT_TOT = zeros(size(EEG_GT_Segmented,1),1);
SampEn_Corrupted_TOT = zeros(size(EEG_GT_Segmented,1),1);
SampEn_Restored_TOT = zeros(size(EEG_GT_Segmented,1),1);
for i = 1:size(EEG_GT_Segmented,1)  % Loop through segments
    EEG_GT_Temp = normalize(EEG_GT_Segmented(i,:),'range');
    EEG_Corrupted_Temp = normalize(EEG_Corrupted_Segmented(i,:),'range');
    EEG_Pred_Temp = normalize(EEG_Pred_Segmented(i,:),'range');
    EEG_Pred_Corrupted_Temp = normalize(EEG_Pred_Corrupted_Segmented(i,:),'range');
    %
    SampEn_Clean_GT = SampEn(EEG_GT_Temp,5,0.1);
    SampEn_Corrupted = SampEn(EEG_Corrupted_Temp,5,0.1);
    SampEn_Restored = SampEn(EEG_Pred_Temp,5,0.1);
    SampEn_Clean_GT_TOT(i,1) = SampEn_Clean_GT;
    SampEn_Corrupted_TOT(i,1) = SampEn_Corrupted;
    SampEn_Restored_TOT(i,1) = SampEn_Restored;
end
SampEn_Clean_GT = mean(SampEn_Clean_GT_TOT,'all',"omitnan");
SampEn_Corrupted = mean(SampEn_Corrupted_TOT,'all',"omitnan");
SampEn_Restored = mean(SampEn_Restored_TOT,'all',"omitnan");
disp(SampEn_Clean_GT);
disp(SampEn_Corrupted);
disp(SampEn_Restored);
clear SampEn_Clean_GT_TOT SampEn_Corrupted_TOT SampEn_Restored_TOT;
%
FuzzyEn_Clean_GT_TOT = zeros(size(EEG_GT_Segmented,1),1);
FuzzyEn_Corrupted_TOT = zeros(size(EEG_GT_Segmented,1),1);
FuzzyEn_Restored_TOT = zeros(size(EEG_GT_Segmented,1),1);
for i = 1:size(EEG_GT_Segmented,1)  % Loop through segments
    EEG_GT_Temp = normalize(EEG_GT_Segmented(i,:),'range');
    EEG_Corrupted_Temp = normalize(EEG_Corrupted_Segmented(i,:),'range');
    EEG_Pred_Temp = normalize(EEG_Pred_Segmented(i,:),'range');
    EEG_Pred_Corrupted_Temp = normalize(EEG_Pred_Corrupted_Segmented(i,:),'range');
    %
    FuzzyEn_Clean_GT = FuzzyEn(EEG_GT_Temp,5,0.1,2);
    FuzzyEn_Corrupted = FuzzyEn(EEG_Corrupted_Temp,5,0.1,2);
    FuzzyEn_Restored = FuzzyEn(EEG_Pred_Temp,5,0.1,2);
    FuzzyEn_Clean_GT_TOT(i,1) = FuzzyEn_Clean_GT;
    FuzzyEn_Corrupted_TOT(i,1) = FuzzyEn_Corrupted;
    FuzzyEn_Restored_TOT(i,1) = FuzzyEn_Restored;
end
FuzzyEn_Clean_GT = mean(FuzzyEn_Clean_GT_TOT,'all',"omitnan");
FuzzyEn_Corrupted = mean(FuzzyEn_Corrupted_TOT,'all',"omitnan");
FuzzyEn_Restored = mean(FuzzyEn_Restored_TOT,'all',"omitnan");
disp(FuzzyEn_Clean_GT);
disp(FuzzyEn_Corrupted);
disp(FuzzyEn_Restored);
clear FuzzyEn_Clean_GT_TOT FuzzyEn_Corrupted_TOT FuzzyEn_Restored_TOT;
%
PermEn_Clean_GT_TOT = zeros(size(EEG_GT_Segmented,1),1);
PermEn_Corrupted_TOT = zeros(size(EEG_GT_Segmented,1),1);
PermEn_Restored_TOT = zeros(size(EEG_GT_Segmented,1),1);
for i = 1:size(EEG_GT_Segmented,1)  % Loop through segments
    EEG_GT_Temp = normalize(EEG_GT_Segmented(i,:),'range');
    EEG_Corrupted_Temp = normalize(EEG_Corrupted_Segmented(i,:),'range');
    EEG_Pred_Temp = normalize(EEG_Pred_Segmented(i,:),'range');
    EEG_Pred_Corrupted_Temp = normalize(EEG_Pred_Corrupted_Segmented(i,:),'range');
    %
    PermEn_Clean_GT = PermEn(EEG_GT_Temp,5);
    PermEn_Corrupted = PermEn(EEG_Corrupted_Temp,5);
    PermEn_Restored = PermEn(EEG_Pred_Temp,5);
    PermEn_Clean_GT_TOT(i,1) = PermEn_Clean_GT;
    PermEn_Corrupted_TOT(i,1) = PermEn_Corrupted;
    PermEn_Restored_TOT(i,1) = PermEn_Restored;
end
PermEn_Clean_GT = mean(PermEn_Clean_GT_TOT,'all',"omitnan");
PermEn_Corrupted = mean(PermEn_Corrupted_TOT,'all',"omitnan");
PermEn_Restored = mean(PermEn_Restored_TOT,'all',"omitnan");
disp(PermEn_Clean_GT);
disp(PermEn_Corrupted);
disp(PermEn_Restored);
clear PermEn_Clean_GT_TOT PermEn_Corrupted_TOT PermEn_Restored_TOT;
%%
[pxx,fx] = pspectrum(EEG_GT,Fs);
[pyy,fy] = pspectrum(EEG_Corrupted,Fs);
[pzz,fz] = pspectrum(EEG_Pred,Fs);
pxx = pow2db(mean(pxx,2));
pyy = pow2db(mean(pyy,2));
pzz = pow2db(mean(pzz,2));
hold on
plot(fx,pxx,'LineWidth',2)
plot(fy,pyy,'LineWidth',2)
plot(fz,pzz,'LineWidth',2)
xlabel('Frequency (Hz)','FontSize',14)
ylabel('PSD (dB/Hz)','FontSize',14)
xlim([1 40])
set(gcf,'color','w');
legend('EEG Ground Truth','EEG Corrupted','EEG Restored');
title('Periodogram Power Spectral Density (PSD) Estimate for EEG Signals','FontSize',16)
hold off
%% Evaluate per Segment (1024)
warning('off');
clear;
clc;
del_SNR_all_trial = [];
for i = 1:23
    subject_num = i;
    directory = append('Results\Segmentation\EMARS4_2\Preds_Sub_',int2str(subject_num),'.h5');
    EEG_GT = h5read(directory,'/EEG_GT'); % Entire ground truth EEG signal (signal without motion artifacts) after removal of baseline drift
    EEG_Pred = squeeze(h5read(directory,'/EEG_Pred')); % EEG signal after removal of motion artifacts
    EEG_MA = h5read(directory,'/EEG_MA'); % Entire EEG Signal with motion artifacts after removal of baseline drift
    %
    directory1 = append('Data\Segmentation\EEG_IMU_',int2str(subject_num),'.mat');
    load(directory1);
    size_ = size(EEG_GT);
    eeg_gt = zeros(size_(1),size_(2)/2);
    eeg_ma = zeros(size_(1),size_(2)/2);
    eeg_pred = zeros(size_(1),size_(2)/2);
    count = 0;
    for j = 1:2:size_(2) % Remove the Even numbered waveforms to expel 50% overlapping affect
        count = count + 1;
        eeg_gt(:,count) = EEG_GT(:,j)*amp_EEG_GT(:,j);
        eeg_ma(:,count) = EEG_MA(:,j)*amp_EEG_MA(:,j);
        %pred_sig = normalize(normalize(EEG_Pred(:,j),'zscore'),'range');
        pred_sig = EEG_Pred(:,j);
        eeg_pred(:,count) = pred_sig*amp_EEG_GT(:,j);
    end
    % Performance parameter Estimation (1. DSNR and 2. Percentage reduction in artifacts (Improvement in correlation)
    size_ = size(eeg_gt);
    del_snr_tot = zeros(1,size_(2));
    corr_af_tot = zeros(2,2,size_(2));
    corr_bf_tot = zeros(2,2,size_(2));
    for j = 1:size_(2)
        del_snr = 10*log10(var(eeg_gt(:,j))/var(eeg_pred(:,j))) - 10*log10(var(eeg_gt(:,j))/var(eeg_ma(:,j)));
        corr_af_tempo = corrcoef(eeg_gt(:,j),eeg_pred(:,j));
        corr_bf_tempo = corrcoef(eeg_gt(:,j),eeg_ma(:,j));
        del_snr_tot(j) = del_snr;
        corr_af_tot(1,1,j) = corr_af_tempo(1,1);
        corr_af_tot(1,2,j) = corr_af_tempo(1,2);
        corr_af_tot(2,1,j) = corr_af_tempo(2,1);
        corr_af_tot(2,2,j) = corr_af_tempo(2,2);
        corr_bf_tot(1,1,j) = corr_bf_tempo(1,1);
        corr_bf_tot(1,2,j) = corr_bf_tempo(1,2);
        corr_bf_tot(2,1,j) = corr_bf_tempo(2,1);
        corr_bf_tot(2,2,j) = corr_bf_tempo(2,2);
    end
    del_snr = mean(del_snr_tot);
    corr_af_tempo(1,1) = mean(corr_af_tot(1,1,:));
    corr_af_tempo(1,2) = mean(corr_af_tot(1,2,:));
    corr_af_tempo(2,1) = mean(corr_af_tot(2,1,:));
    corr_af_tempo(2,2) = mean(corr_af_tot(2,2,:));
    %
    corr_bf_tempo(1,1) = mean(corr_bf_tot(1,1,:));
    corr_bf_tempo(1,2) = mean(corr_bf_tot(1,2,:));
    corr_bf_tempo(2,1) = mean(corr_bf_tot(2,1,:));
    corr_bf_tempo(2,2) = mean(corr_bf_tot(2,2,:));
    %
    corr_imprvmnt_tempo = 100 * (1 - (1 - corr_af_tempo(1,2))/(1 - corr_bf_tempo(1,2)));
    Performance_matix = [del_snr corr_imprvmnt_tempo];
    del_SNR_all_trial = cat(2,del_SNR_all_trial,del_snr);
    fprintf('Trial: %d, Corr_af: %f, Corr_bf: %f, Corr_Improvement: %f, Del_SNR: %f\n',i,corr_af_tempo(1,2),corr_bf_tempo(1,2),corr_imprvmnt_tempo,del_snr);
end
del_SNR_all_trial = del_SNR_all_trial';
%%
figure;
hold on
k = 50;
plot(eeg_gt(:,k)/amp_EEG_MA(:,k));
plot(eeg_ma(:,k)/amp_EEG_MA(:,k));
plot(eeg_pred(:,k)/amp_EEG_MA(:,k));
hold off
xlim([0 1024])
legend('EEG Ground Truth','EEG Motion Corrupted','EEG Estimated');
title('EEG Motion Artifact Removal using EMARS Network');
%% Plotting Prepared Whole Signal per Trial
eeg_ma_tot1 = normalize(eeg_ma_tot,'zscore');
figure;
subplot(3,1,1);
plot(eeg_gt_tot,'k','LineWidth',1);
xlim([0 133120])
title('EEG Ground Truth Signal (Channel 1)','FontSize',12);
subplot(3,1,2);
plot(eeg_ma_tot,'k','LineWidth',1);
xlim([0 133120])
title('EEG Motion Corrupted Signal (Channel 2)','FontSize',12);
subplot(3,1,3);
plot(eeg_pred_tot,'k','LineWidth',1);
xlim([0 133120])
title('Motion Corrected EEG Signal using AG-Operational-CycleGAN','FontSize',12);
sgtitle('EEG Signal Whole Duration (Trial 2)','FontSize',16);
%% Plotting Prepared Whole Signal per Trial with Fake Corrupted ones
% eeg_ma_tot1 = normalize(eeg_ma_tot,'zscore');
figure;
subplot(4,1,1);
plot(eeg_gt_tot,'k','LineWidth',1);
xlim([0 133120])
title('EEG Ground Truth Signal (Channel 1)','FontSize',12);
subplot(4,1,2);
plot(eeg_ma_tot,'k','LineWidth',1);
xlim([0 133120])
title('EEG Motion Corrupted Signal (Channel 2)','FontSize',12);
subplot(4,1,3);
plot(eeg_pred_tot,'k','LineWidth',1);
xlim([0 133120])
title('Motion Corrected EEG Signal using AG-Operational-CycleGAN','FontSize',12);
subplot(4,1,4);
plot(eeg_fake_ma_tot,'k','LineWidth',1);
xlim([0 133120])
title('Fake Corrupted Signal from AG-Operational-CycleGAN','FontSize',12);
sgtitle('EEG Signal Whole Duration (Trial 2)','FontSize',16);
%% Plotting 1024 Length Segments
i = 43;
sig_length = 1024;
figure;
subplot(4,1,1);
A = eeg_gt(:,i);
% A = filtButter(A,256,6,[0.01 40],'bandpass');
% A = NotchFilterIIR(A,50,256,0.5);
plot(A,'k','LineWidth',2);
xlim([0 sig_length])
title('EEG Channel 1','FontSize',12);
subplot(4,1,2);
B = eeg_ma(:,i);
% B = NotchFilterIIR(B,50,256,0.5);
plot(B,'k','LineWidth',2);
xlim([0 sig_length])
title('EEG Channel 2','FontSize',12);
subplot(4,1,3);
C = eeg_pred(:,i);
% C = NotchFilterIIR(C,50,256,0.5);
plot(C,'k','LineWidth',2);
xlim([0 sig_length])
title('AGO-CycleGAN Output','FontSize',12);
subplot(4,1,4);
D = eeg_fake_ma(:,i);
plot(D,'k','LineWidth',2);
xlim([0 sig_length])
title('EEG Fake MA Signals from CycleGAN','FontSize',12);
sgtitle('Sample Motion Corrupted Segment (Trial 2)','FontSize',16);
%%
i = 5;
sig_length = 1024;
figure;
subplot(3,1,1);
A = EEG_GT(:,i);
A = NotchFilterIIR(A,50,256,1);
plot(A,'LineWidth',2);
xlim([0 sig_length])
title('EEG Channel 1 (Ground Truth)','FontSize',14);
ax = gca;
ax.FontSize = 14; 
subplot(3,1,2);
B = EEG_MA(:,i);
B = NotchFilterIIR(B,50,256,1);
plot(B,'LineWidth',2);
xlim([0 sig_length]);
corr_bf = corrcoef(A,B);
ax = gca;
ax.FontSize = 14; 
title(sprintf('EEG Channel 2 (PCC: %0.2f%%)',corr_bf(2)*100),'FontSize',14);
subplot(3,1,3);
C = EEG_Pred(:,i);
C = NotchFilterIIR(C,50,256,1);
plot(C,'LineWidth',2);
xlim([0 sig_length])
corr_af = corrcoef(A,C);
ax = gca;
ax.FontSize = 14; 
title(sprintf('MLMRS-Net Output (PCC: %0.2f%%)',corr_af(2)*100),'FontSize',14);
set(gcf,'color','w');
sgtitle('Sample Clean Segment (Test Fold 3)','FontSize',16);
%%
i = i+1;
sig_length = 1024;
figure;
hold on
A = EEG_GT(:,i);
plot(A,'LineWidth',2);
B = EEG_MA(:,i);
plot(B,'LineWidth',2);
C = EEG_Pred(:,i);
plot(C,'LineWidth',2);
hold off
xlim([0 sig_length])
legend ('EEG Channel 1', 'EEG Channel 2', 'EEG Restored by AGO-CycleGAN');
title('Clean Segment - Test Fold 1','FontSize',12);
%%
x = -1:1/100:1;
y = 10+x;
plot(x,y,'LineWidth',2)
xlim([-1 1]);
ylim([-1 1]);
ax = gca;
ax.FontSize = 18; 
set(gcf,'color','w');