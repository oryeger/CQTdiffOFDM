% main_dir = 'declipping04_01_2026';
main_dir = 'declipping05_01_2026';
subplot(3,1,1);
lim_val = 0.25;
plot_wav_channels('C:\Projects\CQTdiffOFDM\notebooks\experiments\cqt\declipping06_01_2026\original\1.wav')
title('Orig')
ylim([-lim_val lim_val]);
plot_wav_channels('C:\Projects\CQTdiffOFDM\notebooks\experiments\cqt\declipping06_01_2026\clipped_3\1.wav')
title('Clipped')
ylim([-lim_val lim_val]);
plot_wav_channels('C:\Projects\CQTdiffOFDM\notebooks\experiments\cqt\declipping06_01_2026\declipped_3\1.wav')
title('Declipped')
ylim([-lim_val lim_val]);


