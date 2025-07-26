% Plot Periodogram PSD GT vs. MA vs. Pred (Method 1)
hold on;
plot(fx,movmean(pxx_org,6000));
plot(fy,movmean(pyy_org,6000));
plot(fz,movmean(pzz_org,6000));
xlim([1 80])
legend('EEG GT','EEG MA','EEG Estimated');
xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')
title('Periodogram Power Spectral Density (PSD) Estimate for EEG Signals','FontSize',16)
hold off;
%% Plot Power vs. Frequency Plot (Method 2)
eeg_gt_tot_all1 = NotchFilterIIR(eeg_gt_tot_all,50,256,250);
eeg_ma_tot_all1 = NotchFilterIIR(eeg_ma_tot_all,50,256,500);
eeg_pred_tot_all1 = NotchFilterIIR(eeg_pred_tot_all,50,256,250);
[pxx,fx] = pspectrum(eeg_gt_tot_all1,256);
[pyy,fy] = pspectrum(eeg_ma_tot_all1,256);
[pzz,fz] = pspectrum(eeg_pred_tot_all1,256);
hold on
plot(fx,pow2db(pxx),'LineWidth',2)
plot(fy,pow2db(pyy),'LineWidth',2)
plot(fz,pow2db(pzz),'LineWidth',2)
xlabel('Frequency (Hz)','FontSize',14)
ylabel('PSD (dB/Hz)','FontSize',14)
xlim([1 80])
legend('EEG GT','EEG MA','EEG Estimated');
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
%% Plotting 1024 Length Segments
i = 106;
sig_length = 1024;
figure;
subplot(3,1,1);
A = eeg_gt(:,i);
% A = filtButter(A,256,6,[0.01 40],'bandpass');
% A = NotchFilterIIR(A,50,256,0.5);
plot(A,'k','LineWidth',2);
xlim([0 sig_length])
title('EEG Channel 1','FontSize',12);
subplot(3,1,2);
B = eeg_ma(:,i);
% B = NotchFilterIIR(B,50,256,0.5);
plot(B,'k','LineWidth',2);
xlim([0 sig_length])
title('EEG Channel 2','FontSize',12);
subplot(3,1,3);
C = eeg_pred(:,i);
% C = NotchFilterIIR(C,50,256,0.5);
plot(C,'k','LineWidth',2);
xlim([0 sig_length])
title('AG-Operational-CycleGAN Output','FontSize',12);
% subplot(4,1,4);
% D = eeg_fake_ma(:,i);
% plot(D,'LineWidth',2);
% xlim([0 sig_length])
% title('EEG Fake MA Signals from CycleGAN','FontSize',12);
sgtitle('Sample Motion Corrupted Segment (Trial 2)','FontSize',16);