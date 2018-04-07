% This script was used to produce the results of the simulation on BINARY LDA.
% See paper for details on the simulation.
%
% Note: The MATLAB toolbox 'MVPA-Light' is required to run parts of this script.

clear all
addpath ~/git/Fast-Least-Squares/
addpath ~/git/Fast-Least-Squares/simulation

% Set some directories
datadir = '~/data/fast_least_squares/';
figdir = [datadir 'figures/'];
resultsdir = [datadir 'results/'];

% Settings for creating random Gaussian data
nclasses = 2;
prop = 'equal';
varm = 0.1;

% Cross-validation settings
metric = 'acc';  % 'auc' 'acc'
lambda = 0.001;   % Regularisation parameter
reg     = 'shrink'; % 'ridge';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    CROSS-VALIDATION     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nrepeats = 20;     % how often the 
nsamples = [100, 1000];
nfolds = [5 , 10, 20, inf];
nfeatures = round(logspace(1,3,40));


%%% Fast permutations
cfg = [];
cfg.metric  = metric;
% note: if .reg is set to 'shrink', the approximation to shrinkage is
% used (since in the fast LDA approach shrinkage is approximated by
% ridge regularisation)
cfg.reg     = reg;
cfg.lambda  = lambda;
cfg.cv      = 'kfold';
cfg.nperm   = 0;   % do no permutations, only cross-validation

fprintf('\n *** Starting cross-validation:\n')
pause(2)
for rr=1:nrepeats
    fprintf('\n\n------------------\n--- Repeat %d ---\n------------------', rr)

    % Variable saving computation time
    time = zeros(2, numel(nsamples), numel(nfolds), numel(nfeatures));
    
    for nn=1:numel(nsamples)            %%% ---- nn: samples
        fprintf('\n--- NSAMPLES = %d\n', nsamples(nn))
        
        for kk=1:numel(nfolds)                   %%% ---- kk: number of folds for CV
            for ff=1:numel(nfeatures)                %%% ---- ff: features
                if mod(ff,10)==0 , fprintf('%d ',ff), end
                
                % Fix random seed [for replicability]
                seed = rr*numel(nsamples)*numel(nfeatures) + nn*numel(nfeatures) + ff;
                
                %%% Create data
                rng(seed)
                [X, clabel] = simulate_gaussian_data(nsamples(nn), nfeatures(ff), nclasses,prop, varm, 0);
                
                %%% Set k
                if isinf(nfolds(kk))
                    cfg.k       = nsamples(nn);
                else
                    cfg.k       = nfolds(kk);
                end
                
                
                %%% Fast LDA permutations
                rng(seed)
                tic
                perf = fast_least_squares(cfg, X, clabel);
                time(1,nn,kk,ff) = toc;
                
                %%% Standard permutations
                rng(seed)
                tic
                perf2 = standard_lda_traintest(cfg, X, clabel);
                time(2,nn,kk,ff) = toc;
            end
        end
    end
    
    save([resultsdir 'simulation1_binary_LDA_cross-validation_' metric '_rep' num2str(rr)],'time','cfg', ...
        'nsamples','nfolds','nfeatures','nrepeats','perf','perf2')
    
    fprintf('\nFinished ** repetition %d **\n\n\n',rr)
    
end


fprintf('finished\n')

%% Plot results
% mtimes = squeeze(mean(time,1));
%
% close all
% for kk=1:numel(nfolds)
%     subplot(1,numel(nfolds),kk)
%
%     semilogx(nfeatures,squeeze(mtimes(:,kk,:)))
%     legend({'Fast LDA' 'Standard'})
%     if kk<numel(nfolds)
%         title(sprintf('CV (k = %d)',nfolds(kk)))
%     else
%         title('CV (leave-one-out)')
%     end
% end

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      PERMUTATIONS       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Focus on 10-fold cross-validation. As nfeatures, use 100 and 1000. As
% nperms use 100
nrepeats = 20;
nsamples = [100, 1000];
nfeatures = [100, 1000];
nfolds = 10;

% Vary number of permutations
nperm= [100, 1000, 10000];

%%% Fast permutations
cfg = [];
cfg.metric  = metric;
% note: if .reg is set to 'shrink', the approximation to shrinkage is
% used (since in the fast LDA approach shrinkage is approximated by
% ridge regularisation)
cfg.reg     = reg;
cfg.lambda  = lambda;
cfg.cv      = 'kfold';

fprintf('\n *** Starting permutations:\n')
pause(2)
for rr=1:nrepeats
    fprintf('\n\n------------------\n--- Repeat %d ---\n------------------\n', rr)
    
    % Variable saving computation time
    time = zeros(2, numel(nsamples), numel(nperm), numel(nfeatures));
    
    for ss=1:numel(nsamples)   
        fprintf('\nnsamples = %d\n', nsamples(ss))

        for pp=1:numel(nperm)
            fprintf('nperm = %d\nnfeatures = ', nperm(pp))
            for ff=1:numel(nfeatures)                %%% ---- ff: features
                fprintf('%d ',nfeatures(ff))
                
                % Fix random seed [for replicability]
                seed = rr*numel(nperm)*numel(nfeatures) + pp*numel(nfeatures) + ff;
                
                %%% Create data
                rng(seed)
                [X, clabel] = simulate_gaussian_data(nsamples(ss), nfeatures(ff),nclasses,prop, varm, 0);
                
                %%% Set number of permutations
                cfg.nperm       = nperm(pp);
                
                %%% Fast LDA permutations
                rng(seed)
                tic
                perf = fast_least_squares(cfg, X, clabel);
                time(1,ss,pp,ff) = toc;
                
                %%% Standard permutations
                rng(seed)
                tic
                perf2 = standard_lda_traintest(cfg, X, clabel);
                time(2,ss,pp,ff) = toc;
            end
            fprintf('\n')
        end
    end
    save([resultsdir 'simulation1_binary_LDA_permutations_' metric '_rep' num2str(rr)],'time','cfg', ...
        'nsamples','nfolds','nfeatures','nrepeats','nperm','perf','perf2')
    pause(60*2)    % give me a break
end

fprintf('finished\n')


