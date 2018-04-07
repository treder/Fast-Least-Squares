% Calculates null distributions for all subjects in the Wakeman & Henson
% dataset. Uses binary LDA.
%
% Requires: Fieldtrip and the MVPA-Light toolbox.

function WakemanHenson_run_permutations_binary_LDA(ss)

close all
%clear

ival = [-0.5, 1];

% Permutation settings
k = 10;
nperm = 100;

% Path to where the MEG data is stored
loaddir = '/cubric/data/sapmt8/WakemanHenson/preproc/';
resultsdir = '/cubric/data/sapmt8/WakemanHenson/results/';

nSbj = 16;

% Processing times for the anlyses
time1 = zeros(2,1); % method [fast,standard] x  sbj
time2 = zeros(2,1); % method [fast,standard] x  sbj

win = 0.1; % averaging window size in seconds
nwins = floor(1 / win) * ival(2);

% Not all subject numbers exist
sbjs = [1:9, 12:16, 18:19];

% for ss=1%:numel(sbjs)
    
    sbj = sbjs(ss);
    fprintf('\n\n*** Processing subject %d ***\n', sbj)

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
    nchans  = size(dat.trial,2);   % 380 (x 20 nwins = 7600

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
    
%     acc = mv_crossvalidate([], X, clabel);
    
    %% Permutations 
    
    %%% Fast permutations
    cfg = [];
    cfg.metric  = 'acc'; % 'auc';
    cfg.reg     = 'shrink';
    cfg.lambda  = 0.001;
    cfg.cv      = 'kfold';
    cfg.k       = k;
    cfg.nperm   = nperm;
    cfg.repeat  = 1;
    
    rng(sbj)
    fprintf('Fast LDA permutations spatio-temporal features...\n')
    tic
    fast_lda_permutations(cfg, X, clabel);
    time1(1) = toc;
    
    %%% Classic permutations
    rng(sbj)
    fprintf('Classical permutations spatio-temporal features...\n')
    tic
    classic_lda_permutations(cfg, X, clabel);
    time1(2) = toc;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ANALYSIS 2: classification across time %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    timeidx = find(dat.time >= ival(1) & dat.time < ival(2) );
    step = 10;
    
    rng(sbj)
    fprintf('Fast LDA permutations [classification across time]...\n')
    tic
    for tt=1:step:numel(timeidx)
        fast_lda_permutations(cfg, squeeze(dat.trial(:,:,timeidx(tt))), clabel);
    end
    time2(1) = toc;
    
    %%% Classic permutations
    rng(sbj)
    fprintf('Classical permutations [classification across time]...\n')
    tic
    for tt=1:step:numel(timeidx)
    	classic_lda_permutations(cfg, squeeze(dat.trial(:,:,timeidx(tt))), clabel);
    end
    time2(2) = toc;
    
    fprintf('Finished subject %d.\n\n', sbj)
    
    %% Save data
    save([resultsdir 'WakemanHenson_binary_LDA_sbj' num2str(ss)],'time1','time2','cfg', ...
        'ntrials','nchans','clabel','k','nperm','win','nwins')


% end

fprintf('Finished all.\n')