%% generate_ofdm_maestro_like.m
% Writes outputs ONLY to: <pwd>/<YYYY-mm-dd>/
%   ofdm_maestro_like_NfftXXXX_MOD_clean.wav
%
% Notes:
% - Saves ONLY the CLEAN wav (no clipped, no meta).
% - Builds a clipped version in memory, then computes FFT-domain EVM
%   between CLEAN and CLIPPED.
% - EVM is measured ONLY on the DATA subcarriers (no guards).

%clear; clc;
% rng(42, 'twister');

%% ---------------- USER SETTINGS ----------------
% main_dir = 'C:\Projects\CQTdiffOFDM\examples\data_dir\';
% example_wav = 'MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_02_Track02_wav.wav';

main_dir = 'C:\Projects\CQTdiffOFDM\examples\maestro-v3.0.0\2004\';
example_wav = 'MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav';
exampleWavPath = [main_dir, example_wav];

outPrefix      = 'ofdm_maestro_like';
durationSec    = 3;

cfg.Nfft        = 2048;     % even
cfg.cpLen       = round(9/128*cfg.Nfft); % can be 0
cfg.dataRatio   = 0.8;      % ~80% data / 20% guards (positive bins)
cfg.modType     = '16QAM';  % 'QPSK'|'16QAM'|'64QAM'|'256QAM'
cfg.useContiguousDataBlock = true;

cfg.cleanPeakTarget = 0.98;

cfg.channelMode = 'match';             % 'match'|'mono'|'stereo'
cfg.multiChannelContent = 'duplicate'; % 'duplicate'|'independent'

% -------- clipping for in-memory analysis (not saved) --------
cfg.hardClipEnable = true;
cfg.hardClipThresh = 0.3;
%% ------------------------------------------------

try
    %% Sanity checks
    if exist(exampleWavPath, 'file') ~= 2
        error('Input WAV not found:\n  %s', exampleWavPath);
    end

    %% Read example WAV header
    info = audioinfo(exampleWavPath);
    Fs = info.SampleRate;

    if isfield(info, 'BitsPerSample') && ~isempty(info.BitsPerSample)
        bits = info.BitsPerSample;
    else
        bits = 16;
    end

    switch lower(cfg.channelMode)
        case 'match'
            nCh = info.NumChannels;
        case 'mono'
            nCh = 1;
        case 'stereo'
            nCh = 2;
        otherwise
            error('cfg.channelMode must be match|mono|stereo');
    end

    %% Generate CLEAN OFDM
    Nsamp = round(durationSec * Fs);

    if nCh == 1
        yClean = build_real_ofdm(cfg, Fs, durationSec);
    else
        if strcmpi(cfg.multiChannelContent, 'duplicate')
            mono = build_real_ofdm(cfg, Fs, durationSec);
            yClean = repmat(mono, 1, nCh);
        elseif strcmpi(cfg.multiChannelContent, 'independent')
            yClean = zeros(Nsamp, nCh);
            for c = 1:nCh
                yClean(:,c) = build_real_ofdm(cfg, Fs, durationSec);
            end
        else
            error('cfg.multiChannelContent must be duplicate|independent');
        end
    end

    %% Scale CLEAN to known peak
    yClean = scale_to_peak(yClean, cfg.cleanPeakTarget);

    %% Create CLIPPED version in memory (not saved)
    yClip = yClean;
    if cfg.hardClipEnable
        yClip = hardclip(yClip, cfg.hardClipThresh);
    end

    %% Output path (ONLY in outDir) - include FFT size and modulation
    tag = sprintf('Nfft%d_%s', cfg.Nfft, upper(cfg.modType));
    cleanWav = fullfile(outDir, sprintf('%s_%s_clean.wav', outPrefix, tag));

    %% Write WAV (CLEAN only)
    audiowrite(cleanWav, yClean, Fs, 'BitsPerSample', bits);
    [yClean_wav, Fs_wav] = audioread(cleanWav);


    %% EVM analysis (per-channel) using a function
    emptyEvm = struct( ...
        'evmRms', [], ...
        'evmDb', [], ...
        'nSymUsed', [], ...
        'nUsedSubcarriers', [], ...
        'usedBins', [], ...
        'mode', '' );

    evm = struct();
    evm.perCh = repmat(emptyEvm, 1, nCh);

    for ch = 1:nCh
        if nCh == 1
            xC = yClean(:);
            xK = yClip(:);
        else
            xC = yClean(:,ch);
            xK = yClip(:,ch);
        end

        evm.perCh(ch) = compute_evm_ofdm(cfg, xC, xK);
    end

    %% Print summary
    fprintf('\nExample WAV: Fs=%d | ch=%d | bits=%d\n', info.SampleRate, info.NumChannels, bits);
    fprintf('Output WAV : Fs=%d | ch=%d | bits=%d | dur=%.2fs\n', Fs, nCh, bits, durationSec);
    fprintf('OFDM      : Nfft=%d | cpLen=%d | dataRatio=%.2f | mod=%s\n', cfg.Nfft, cfg.cpLen, cfg.dataRatio, cfg.modType);
    fprintf('PAPR      : clean=%.2f dB | clipped=%.2f dB\n', papr_db(yClean), papr_db(yClip));
    fprintf('Clip      : enabled=%d | thr=%.2f\n', cfg.hardClipEnable, cfg.hardClipThresh);

    for ch = 1:nCh
        fprintf('EVM (ch %d) [data-only]: %.6f (%.2f dB) | dataSC=%d | nSym=%d\n', ...
            ch, evm.perCh(ch).evmRms, evm.perCh(ch).evmDb, ...
            evm.perCh(ch).nUsedSubcarriers, evm.perCh(ch).nSymUsed);
    end

    fprintf('\nWrote:\n  %s\n', cleanWav);

catch ME
    fprintf(2, '\nERROR: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
    end
end

save cfg_save cfg


%% ================= Helper functions =================
function y = build_real_ofdm(cfg, Fs, durationSec)
Nfft  = cfg.Nfft;
cpLen = cfg.cpLen;

if mod(Nfft,2) ~= 0
    error('Nfft must be even');
end

switch upper(cfg.modType)
    case 'QPSK',   M = 4;
    case '16QAM',  M = 16;
    case '64QAM',  M = 64;
    case '256QAM', M = 256;
    otherwise
        error('modType must be QPSK/16QAM/64QAM/256QAM');
end

posBins = get_data_pos_bins(cfg);   % data bins only (positive side)
negBins = Nfft - posBins + 2;

symLen      = Nfft + cpLen;
NsampTarget = round(durationSec * Fs);
Nsym        = ceil(NsampTarget / symLen);

y = zeros(Nsym * symLen, 1);

for s = 1:Nsym
    X = zeros(Nfft,1);

    NdataPos = numel(posBins);
    dataIdx = randi([0 M-1], NdataPos, 1);
    dataSym = qammod(dataIdx, M, 'UnitAveragePower', true);

    X(1) = 0;
    X(Nfft/2 + 1) = 0;
    X(posBins) = dataSym;
    X(negBins) = conj(dataSym);

    x = ifft(X, Nfft, 'symmetric');

    if cpLen > 0
        x = [x(end-cpLen+1:end); x];
    end

    i0 = (s-1)*symLen + 1;
    y(i0:i0+symLen-1) = x;
end

y = y(1:NsampTarget);
end

function posBins = get_data_pos_bins(cfg)
Nfft = cfg.Nfft;
NposAvail = Nfft/2 - 1;                 % bins 2..Nfft/2
NdataPos  = floor(cfg.dataRatio * NposAvail);
NguardPos = NposAvail - NdataPos;

if cfg.useContiguousDataBlock
    guardLow = floor(NguardPos/2);
    posBins  = (2 + guardLow) : (1 + guardLow + NdataPos);
else
    usable  = 2:(Nfft/2);
    idx     = round(linspace(1, numel(usable), NdataPos));
    posBins = usable(idx);
end
posBins = posBins(:);
end

function y = scale_to_peak(x, peakTarget)
pk = max(abs(x(:))) + eps;
y  = (peakTarget / pk) * x;
end

function y = hardclip(x, A)
y = min(max(x, -A), A);
end

function p = papr_db(x)
pk = max(abs(x(:))) + eps;
r  = rms(x(:)) + eps;
p = 20*log10(pk/r);
end

function out = compute_evm_ofdm(cfg, xClean, xTest)
% compute_evm_ofdm
% Computes FFT-domain EVM between a reference OFDM waveform (xClean)
% and a test waveform (xTest) under the SAME OFDM parameters in cfg.
%
% IMPORTANT: EVM is measured ONLY on the DATA subcarriers (no guards).
%
% EVM = sqrt( mean(|X_test - X_ref|^2) / mean(|X_ref|^2) )
% where X are the FFT-domain symbols on the data subcarriers.

Nfft  = cfg.Nfft;
cpLen = cfg.cpLen;
symLen = Nfft + cpLen;

L = min(numel(xClean), numel(xTest));
xClean = xClean(1:L);
xTest  = xTest(1:L);

nSym = floor(L / symLen);
if nSym < 1
    error('Not enough samples for even 1 OFDM symbol. Increase durationSec or reduce Nfft/cpLen.');
end

% Reshape into symbols
xc = reshape(xClean(1:nSym*symLen), symLen, nSym);
xt = reshape(xTest(1:nSym*symLen),  symLen, nSym);

% Remove CP
if cpLen > 0
    xc_nocp = xc(cpLen+1:end, :);
    xt_nocp = xt(cpLen+1:end, :);
else
    xc_nocp = xc;
    xt_nocp = xt;
end

% FFT along time dimension (rows)
Xc = fft(xc_nocp, Nfft, 1);
Xt = fft(xt_nocp, Nfft, 1);

% Use ONLY the data bins (positive side) - excludes guards, DC, Nyquist
usedBins = get_data_pos_bins(cfg);
modeStr  = 'dataOnly';

ref  = Xc(usedBins, :);
meas = Xt(usedBins, :);



err = meas - ref;

refPow = mean(abs(ref(:)).^2) + eps;
errPow = mean(abs(err(:)).^2);

evmRms = sqrt(errPow / refPow);
evmDb  = 20*log10(evmRms + eps);

out = struct();
out.evmRms = evmRms;
out.evmDb  = evmDb;
out.nSymUsed = nSym;
out.nUsedSubcarriers = numel(usedBins);
out.usedBins = usedBins;
out.mode = modeStr;

figure
plot(Xc(usedBins, :),'x');
title(['Clip=',num2str(cfg.hardClipThresh),', EVM = ',num2str(round(out.evmDb)),'dB'])
grid;
shg;

figure
plot(Xt(usedBins, :),'x');
title(['Clip=',num2str(cfg.hardClipThresh),', EVM = ',num2str(round(out.evmDb)),'dB'])
grid;
shg;

end
