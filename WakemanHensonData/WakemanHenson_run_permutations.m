% Calculates null distributions for all subjects in the Wakeman & Henson
% dataset, for each time point in the trial separately.
%
% Requires: Fieldtrip and the MVPA-Light toolbox.

close all
clear

ival = [-0.5, 1.5];

% Permutation settings
K = 10;
nperm = 1000;

% Path to where the MEG data is stored
loaddir = 'D:/data/WakemanHenson/preproc/';

nSbj = 16;

% Processing times for the anlyses
time1 = zeros(2,nSbj); % method [fast,standard] x  sbj
time2 = zeros(2,nSbj); % method [fast,standard] x  sbj

win = 0.05; % averaging window size in seconds
nwins = round(1 / win) * diff(ival);

for sbj=1:nSbj
    
    %% Load data
    load([loaddir sprintf('sub%03d',sbj)])
    
    % --- Class labels ---
    % From the README file, we know that
    % Trigger codes 5, 6, 7, 13, 14, 15 code for faces,
    % while codes 17, 18, 19 code for scrambled images.
    clabel = dat.trialinfo;
    clabel(ismember(clabel,[ 5, 6, 7, 13, 14, 15])) = 1;  % faces
    clabel(ismember(clabel,[ 17,18,19])) = 2;  % scrambled
  
    % Put the data into handy 3d matrix
    cfg             = [];
    cfg.keeptrials  = 'yes';
    dat = ft_timelockanalysis(cfg, dat);

    ntrials = size(dat.trial,1);
    nchans  = size(dat.trial,2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ANALYSIS 1: spatio-temporal features  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Feature extraction
    % average voltage in successive 50 ms steps
    X = zeros(ntrials, nchans, nwins);
    
    for ii=1:ntrials
        timeidx = 1;
        for tt=1:nwins
            sam = find(dat.time >= (tt-1)*win & dat.time < tt*win);
            X(ii,:,tt) = mean(dat.trial(ii,:,sam),3);
            timeidx= timeidx+1;
        end
    end
    

    % Reshape into 2D matrix by concatenating the time windows as features
    X = reshape(X, ntrials, []);
    
    acc = mv_crossvalidate([], X, clabel);
    
    %% Permutations 
    
    %%% Fast permutations
    cfg = [];
    cfg.metric  = 'acc'; % 'auc';
    cfg.reg     = 'shrink';
    cfg.lambda  = 0.001;
    cfg.cv      = 'kfold';
    cfg.k       = K;
    cfg.nperm   = nperm;
    
    rng(sbj)
    fprintf('Fast LDA permutations...\n')
    tic
    fast_lda_permutations(cfg, X, clabel);
    time1(1,sbj) = toc;
    
    %%% Classic permutations
    rng(sbj)
    fprintf('Classical permutations...\n')
    tic
    classic_lda_permutations(cfg, X, clabel);
    time1(2,sbj) = toc;
    
    fprintf('Finished subject %d.\n\n', sbj)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ANALYSIS 2: classification across time %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    timeidx = find(dat.time >= ival(1) & dat.time < ival(2) );
    
    rng(sbj)
    fprintf('Fast LDA permutations...\n')
    tic
    for tt=1:numel(timeidx)
        fast_lda_permutations(cfg, squeeze(dat.trial(:,:,timeidx(tt))), clabel);
    end
    time2(1,sbj) = toc;
    
    %%% Classic permutations
    rng(sbj)
    fprintf('Classical permutations...\n')
    tic
    for tt=1:numel(timeidx)
    	classic_lda_permutations(cfg, squeeze(dat.trial(:,:,timeidx(tt))), clabel);
    end

    time2(2,sbj) = toc;
    
    fprintf('Finished subject %d.\n\n', sbj)

end
