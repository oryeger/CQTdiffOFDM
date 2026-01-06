function plot_wav_channels(filename)
% PLOT_WAV_CHANNELS  Plot waveform of a WAV file, channels separately
%
% Usage:
%   plot_wav_channels('myfile.wav')
%
% - Supports mono and stereo WAV files
% - Time axis in seconds

    if exist(filename, 'file') ~= 2
        error('File not found: %s', filename);
    end

    [x, Fs] = audioread(filename);
    Fs

    Nsamp = size(x,1);
    nCh   = size(x,2);

    t = (0:Nsamp-1) / Fs;

    figure('Name', filename, 'NumberTitle', 'off');

    if nCh == 1
        % Mono
        plot(t, x(:,1));
        xlabel('Time (s)');
        ylabel('Amplitude');
        title('Mono channel');
        grid on;
    else
        % Multi-channel (e.g. stereo)
        for ch = 1:nCh
            subplot(nCh,1,ch);
            plot(t, x(:,ch));
            ylabel(sprintf('Ch %d', ch));
            grid on;

            if ch == 1
                title('Waveform (channels separated)');
            end
            if ch == nCh
                xlabel('Time (s)');
            end
        end
    end
end
