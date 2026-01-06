function out = wavs_to_ofdm_and_evm(cfg, wavA, wavB, wavC, varargin)
% wavs_to_ofdm_and_evm
% Reverse of your build_real_ofdm chain:
%   WAV time samples -> (optional sync) -> CP remove -> FFT -> extract DATA bins
% Then computes EVM between the WAV-derived OFDM grids on DATA bins only.
%
% Supports:
%   - 2-file mode: A (clean) vs B (test)
%   - 3-file mode: A (clean) vs B (test1) AND A (clean) vs C (test2)
%
% INPUTS:
%   cfg.Nfft, cfg.cpLen, cfg.dataRatio, cfg.useContiguousDataBlock
%   wavA, wavB, wavC: either
%       - file paths (char/string), OR
%       - time-domain vectors (Nx1) or matrices (NxC)
%
% OPTIONAL name-value:
%   'Fs'            : required if wav inputs are vectors (not files)
%   'Sync'          : true/false (default true) - CP-based coarse timing
%   'MaxSyncSearch' : samples to search for sync (default 3*(Nfft+cpLen))
%   'Plot'          : true/false (default false)
%
% OUTPUT (struct):
%   out.Fs
%   out.usedBins
%   out.A.grid, out.B.grid, out.C.grid (if provided)
%   out.perCh(ch).B.evmRms/evmDb and out.perCh(ch).C.evmRms/evmDb (if provided)
%   out.perCh(ch).startA/startB/startC

% ------------------ handle optional wavC ------------------
% Allow calling with only (cfg, wavA, wavB, ...) i.e. old interface.
if nargin < 4
    wavC = [];
end

% If user called old style: wavs_to_ofdm_and_evm(cfg, A, B, 'Plot',true)
% then wavC is actually a char/string parameter name.
if (ischar(wavC) || isstring(wavC)) && any(strcmpi(string(wavC), ["Fs","Sync","MaxSyncSearch","Plot"]))
    % shift: treat wavC as the first name-value key
    varargin = [{wavC}, varargin];
    wavC = [];
end

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

hasC = ~isempty(wavC);
if hasC
    [xC, FsC] = load_wav_or_array(wavC, Fs_in);
end

if FsA ~= FsB
    error('Sample-rate mismatch: FsA=%g, FsB=%g. Resample before calling.', FsA, FsB);
end
if hasC && (FsA ~= FsC)
    error('Sample-rate mismatch: FsA=%g, FsC=%g. Resample before calling.', FsA, FsC);
end
Fs = FsA;

% Ensure column-major [N x C]
if isrow(xA), xA = xA(:); end
if isrow(xB), xB = xB(:); end
if hasC && isrow(xC), xC = xC(:); end

nCh = max(size(xA,2), size(xB,2));
if hasC
    nCh = max(nCh, size(xC,2));
end

if size(xA,2) ~= nCh, xA = repmat(xA(:,1), 1, nCh); end
if size(xB,2) ~= nCh, xB = repmat(xB(:,1), 1, nCh); end
if hasC && size(xC,2) ~= nCh, xC = repmat(xC(:,1), 1, nCh); end

usedBins = get_data_pos_bins(cfg);
symLen   = cfg.Nfft + cfg.cpLen;

if isempty(maxSearch)
    maxSearch = 3*symLen; % default: search a few symbols
end

% ---- demod both/three wavs per-channel ----
gridA = [];
gridB = [];
gridC = [];

% perCh struct template
emptyBC = struct('evmRms',[],'evmDb',[],'nSymUsed',[],'nUsedSubcarriers',[]);
empty = struct('B', emptyBC, 'C', emptyBC, 'startA',[],'startB',[],'startC',[]);

out = struct();
out.Fs = Fs;
out.usedBins = usedBins;
out.perCh = repmat(empty, 1, nCh);

for ch = 1:nCh
    a = xA(:,ch);
    b = xB(:,ch);
    if hasC
        c = xC(:,ch);
    end

    if doSync
        startA = coarse_sync_cp(cfg, a, maxSearch);
        startB = coarse_sync_cp(cfg, b, maxSearch);
        if hasC
            startC = coarse_sync_cp(cfg, c, maxSearch);
        else
            startC = [];
        end
    else
        startA = 1;
        startB = 1;
        if hasC, startC = 1; else, startC = []; end
    end

    [GA, nSymA] = demod_ofdm_grid(cfg, a, startA, usedBins);
    [GB, nSymB] = demod_ofdm_grid(cfg, b, startB, usedBins);

    if hasC
        [GC, nSymC] = demod_ofdm_grid(cfg, c, startC, usedBins);
        nSym = min([nSymA, nSymB, nSymC]);
        GC = GC(:,1:nSym);
    else
        nSym = min(nSymA, nSymB);
    end

    GA = GA(:,1:nSym);
    GB = GB(:,1:nSym);

    % allocate output cubes on first iter
    if isempty(gridA)
        gridA = zeros(numel(usedBins), nSym, nCh);
        gridB = zeros(numel(usedBins), nSym, nCh);
        if hasC
            gridC = zeros(numel(usedBins), nSym, nCh);
        end
    end
    gridA(:, :, ch) = GA;
    gridB(:, :, ch) = GB;
    if hasC
        gridC(:, :, ch) = GC;
    end

    % ---- EVM on DATA bins only ----
    % B vs A
    [evmRmsB, evmDbB] = evm_from_grids(GA, GB);

    out.perCh(ch).B.evmRms = evmRmsB;
    out.perCh(ch).B.evmDb  = evmDbB;
    out.perCh(ch).B.nSymUsed = nSym;
    out.perCh(ch).B.nUsedSubcarriers = numel(usedBins);

    % C vs A (if provided)
    if hasC
        [evmRmsC, evmDbC] = evm_from_grids(GA, GC);

        out.perCh(ch).C.evmRms = evmRmsC;
        out.perCh(ch).C.evmDb  = evmDbC;
        out.perCh(ch).C.nSymUsed = nSym;
        out.perCh(ch).C.nUsedSubcarriers = numel(usedBins);
    end

    out.perCh(ch).startA = startA;
    out.perCh(ch).startB = startB;
    out.perCh(ch).startC = startC;

    if doPlot
        %% ---- Clean constellation (GA) ----
        figure;
        plot(GA, 'x');
        grid on;
        
        % ---- zoom out slightly ----
        re = real(GA(:));
        im = imag(GA(:));
        
        rmax = max(abs(re));
        imax = max(abs(im));
        margin = 1.4;
        
        xlim(margin * [-rmax, rmax]);
        ylim(margin * [-imax, imax]);
        
        title('Orig');
        xlabel('I');
        ylabel('Q');
        shg;

        %% ---- Test1 constellation (GB) ----
        figure;
        plot(GB, 'x');
        grid on; axis equal;
        title(sprintf('Clipped, EVM = %.2f dB',evmDbB));
        xlabel('I'); ylabel('Q'); shg;

        %% ---- Test2 constellation (GC) ----
        if hasC
            figure;
            plot(GC, 'x');
            grid on; axis equal;
            title(sprintf('Declipped, EVM = %.2f dB',evmDbC));
            xlabel('I'); ylabel('Q'); shg;
        end
    end
end

out.A.grid = gridA;
out.B.grid = gridB;
if hasC
    out.C.grid = gridC;
end
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
x = double(x);
end

function [evmRms, evmDb] = evm_from_grids(Gref, Gtest)
err = Gtest - Gref;
refPow = mean(abs(Gref(:)).^2) + eps;
errPow = mean(abs(err(:)).^2);
evmRms = sqrt(errPow / refPow);
evmDb  = 20*log10(evmRms + eps);
end

function [G, nSym] = demod_ofdm_grid(cfg, x, startIdx, usedBins)
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

K = max(1, min(maxSearch, L - symLen));
scores = zeros(K,1);

for k = 1:K
    seg = x(k:k+symLen-1);

    cp   = seg(1:cpLen);
    tail = seg(end-cpLen+1:end);

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
