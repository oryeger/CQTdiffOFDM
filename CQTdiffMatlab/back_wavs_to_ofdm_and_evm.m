function out = wavs_to_ofdm_and_evm(cfg, wavA, wavB, varargin)
% wavs_to_ofdm_and_evm
% Reverse of your build_real_ofdm chain:
%   WAV time samples -> (optional sync) -> CP remove -> FFT -> extract DATA bins
% Then computes EVM between the two WAV-derived OFDM grids on DATA bins only.
%
% INPUTS:
%   cfg.Nfft, cfg.cpLen, cfg.dataRatio, cfg.useContiguousDataBlock
%   wavA, wavB: either
%       - file paths (char/string), OR
%       - time-domain vectors (Nx1) or matrices (NxC)
%
% OPTIONAL name-value:
%   'Fs'            : required if wavA/wavB are vectors (not files)
%   'Sync'          : true/false (default true) - CP-based coarse timing
%   'MaxSyncSearch' : samples to search for sync (default 3*(Nfft+cpLen))
%   'Plot'          : true/false (default false)
%
% OUTPUT (struct):
%   out.Fs
%   out.usedBins
%   out.A.grid, out.B.grid   : [NdataSubcarriers x NsymUsed x Nch] complex
%   out.perCh(ch).evmRms / evmDb / nSymUsed / nUsedSubcarriers
%   out.perCh(ch).startA / startB (sync offsets)
%
% NOTE:
% - If the WAVs are already perfectly aligned (like your generated ones),
%   you can set 'Sync', false.

p = inputParser;
p.addParameter('Fs', [], @(x) isempty(x) || (isscalar(x) && x>0));
p.addParameter('Sync', true, @(x) islogical(x) && isscalar(x));
p.addParameter('MaxSyncSearch', [], @(x) isempty(x) || (isscalar(x) && x>=0));
p.addParameter('Plot', false, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
Fs_in = p.Results.Fs;
doSync = p.Results.Sync;
maxSearch = p.Results.MaxSyncSearch;
doPlot = p.Results.Plot;

% ---- load wavs (or accept vectors) ----
[xA, FsA] = load_wav_or_array(wavA, Fs_in);
[xB, FsB] = load_wav_or_array(wavB, Fs_in);

if FsA ~= FsB
    error('Sample-rate mismatch: FsA=%g, FsB=%g. Resample before calling.', FsA, FsB);
end
Fs = FsA;

% Ensure column-major [N x C]
if isrow(xA), xA = xA(:); end
if isrow(xB), xB = xB(:); end

nCh = max(size(xA,2), size(xB,2));
if size(xA,2) ~= nCh, xA = repmat(xA(:,1), 1, nCh); end
if size(xB,2) ~= nCh, xB = repmat(xB(:,1), 1, nCh); end

usedBins = get_data_pos_bins(cfg);
symLen   = cfg.Nfft + cfg.cpLen;

if isempty(maxSearch)
    maxSearch = 3*symLen; % default: search a few symbols
end

% ---- demod both wavs per-channel ----
gridA = [];
gridB = [];

empty = struct('evmRms',[],'evmDb',[],'nSymUsed',[],'nUsedSubcarriers',[], ...
               'startA',[],'startB',[]);

out = struct();
out.Fs = Fs;
out.usedBins = usedBins;
out.perCh = repmat(empty, 1, nCh);

for ch = 1:nCh
    a = xA(:,ch);
    b = xB(:,ch);

    if doSync
        startA = coarse_sync_cp(cfg, a, maxSearch);
        startB = coarse_sync_cp(cfg, b, maxSearch);
    else
        startA = 1;
        startB = 1;
    end

    [GA, nSymA] = demod_ofdm_grid(cfg, a, startA, usedBins);
    [GB, nSymB] = demod_ofdm_grid(cfg, b, startB, usedBins);

    nSym = min(nSymA, nSymB);
    GA = GA(:,1:nSym);
    GB = GB(:,1:nSym);

    % allocate output cubes on first iter
    if isempty(gridA)
        gridA = zeros(numel(usedBins), nSym, nCh);
        gridB = zeros(numel(usedBins), nSym, nCh);
    end
    gridA(:, :, ch) = GA;
    gridB(:, :, ch) = GB;

    % ---- EVM on DATA bins only ----
    err = GB - GA;
    refPow = mean(abs(GA(:)).^2) + eps;
    errPow = mean(abs(err(:)).^2);

    evmRms = sqrt(errPow / refPow);
    evmDb  = 20*log10(evmRms + eps);

    out.perCh(ch).evmRms = evmRms;
    out.perCh(ch).evmDb  = evmDb;
    out.perCh(ch).nSymUsed = nSym;
    out.perCh(ch).nUsedSubcarriers = numel(usedBins);
    out.perCh(ch).startA = startA;
    out.perCh(ch).startB = startB;

    if doPlot
        %% ---- Clean constellation (GA) ----
        figure;
        plot(GA, 'x');
        grid on;
        axis equal;
        
        title(sprintf('Ch %d | CLEAN', ch));
        xlabel('I');
        ylabel('Q');
        shg;
        

        %% ---- Test constellation (GB) ----
        figure;
        plot(GB, 'x');
        grid on;
        axis equal;
        
        title(sprintf('Ch %d | TEST | EVM = %.2f dB | startA=%d startB=%d', ...
            ch, evmDb, startA, startB));
        xlabel('I');
        ylabel('Q');
        shg;

    end
end

out.A.grid = gridA;
out.B.grid = gridB;
end

%% ===================== helpers =====================

function [x, Fs] = load_wav_or_array(w, Fs_in)
if ischar(w) || isstring(w)
    w = char(w);
    if exist(w,'file') ~= 2
        error('WAV file not found: %s', w);
    end
    [x, Fs] = audioread(w);
else
    x = w;
    if isempty(Fs_in)
        error('If wav input is an array, you must pass ''Fs'', sampleRate.');
    end
    Fs = Fs_in;
end
% Force double
x = double(x);
end

function [G, nSym] = demod_ofdm_grid(cfg, x, startIdx, usedBins)
% Extract OFDM symbols -> FFT -> return data bins grid [Ndata x Nsym]
Nfft  = cfg.Nfft;
cpLen = cfg.cpLen;
symLen = Nfft + cpLen;

x = x(startIdx:end);

nSym = floor(numel(x)/symLen);
if nSym < 1
    G = complex(zeros(numel(usedBins),0));
    nSym = 0;
    return;
end

X = reshape(x(1:nSym*symLen), symLen, nSym);

% remove CP
if cpLen > 0
    X = X(cpLen+1:end, :);
end

% FFT
XF = fft(X, Nfft, 1);

% data bins only
G = XF(usedBins, :);
end

function startIdx = coarse_sync_cp(cfg, x, maxSearch)
% Very simple CP-based timing:
% choose start that maximizes correlation between CP and end-of-symbol.
Nfft  = cfg.Nfft;
cpLen = cfg.cpLen;
symLen = Nfft + cpLen;

if cpLen <= 0
    startIdx = 1;
    return;
end

L = min(numel(x), maxSearch + symLen);
if L < symLen
    startIdx = 1;
    return;
end

x = x(1:L);

% Search candidate starts
K = max(1, min(maxSearch, L - symLen));
scores = zeros(K,1);

for k = 1:K
    seg = x(k:k+symLen-1);

    cp   = seg(1:cpLen);
    tail = seg(end-cpLen+1:end);

    % normalized correlation magnitude
    scores(k) = abs(cp' * tail) / (norm(cp)*norm(tail) + eps);
end

[~, best] = max(scores);
startIdx = best;
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
