% This script analyses the Wakeman and Henson LDA simulation results.
clear 
close all

% Set some directories
datadir = '~/data/fast_least_squares/';
figdir = [datadir 'figures/'];
resultsdir = [datadir 'results/'];

nsbj = 16;

dat = cell(nsbj, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         BINARY  LDA           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for sbj=1:nsbj
    dat{sbj} = load([resultsdir 'WakemanHenson_binary_LDA_sbj' num2str(sbj)]);
    dat{sbj}.sbj = sbj;
   
end

% Convert cell array to struct array
dat = [dat{:}];

% dat.time is of dimensions [method x nsamples x nfolds x nfeatures]

fprintf('finished\n')

%% Compute relative efficiency
% It is defined as log10 of the fraction computation time(standard) / computation
% time (update rule)

efficiency = zeros(sbj, 2);

for sbj=1:sbj
    efficiency(sbj,1) = dat(sbj).time1(2,:,:,:) ./ dat(sbj).time1(1,:,:,:);
    efficiency(sbj,2) = dat(sbj).time2(2,:,:,:) ./ dat(sbj).time2(1,:,:,:);
end

mean_efficiency = squeeze(mean(efficiency,1));

log_efficiency = log10(efficiency);
mean_log_efficiency = log10(mean_efficiency);

trials = [dat.ntrials];
features = [10,1] * dat(1).nchans;

save([resultsdir 'WakemanHenson_binary_LDA'], 'mean_efficiency',...
    'log_efficiency','mean_log_efficiency','trials','-v6');

% convert to table
values = cell(2,1); 
values{2} = features;
T = ndarray2table(log_efficiency, {'Subject' 'Features'}, values);
writetable(T,[resultsdir 'table_WakemanHenson_binary_LDA.csv'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       MULTI-CLASS LDA         %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dat = cell(nsbj, 1);

for sbj=1:nsbj
    dat{sbj} = load([resultsdir 'WakemanHenson_multiclass_LDA_sbj' num2str(sbj)]);
    dat{sbj}.sbj = sbj;
   
end

% Convert cell array to struct array
dat = [dat{:}];

% dat.time is of dimensions [method x nsamples x nfolds x nfeatures]

fprintf('finished\n')

%% Compute relative efficiency
% It is defined as log10 of the fraction computation time(standard) / computation
% time (update rule)

efficiency = zeros(sbj, 2);

for sbj=1:sbj
    efficiency(sbj,1) = dat(sbj).time1(2,:,:,:) ./ dat(sbj).time1(1,:,:,:);
    efficiency(sbj,2) = dat(sbj).time2(2,:,:,:) ./ dat(sbj).time2(1,:,:,:);
end

mean_efficiency = squeeze(mean(efficiency,1));

log_efficiency = log10(efficiency);
mean_log_efficiency = log10(mean_efficiency);

trials = [dat.ntrials];
features = [1/dat(1).win, 1] * dat(1).nchans; % 1/dat(1).win = number of windows

save([resultsdir 'WakemanHenson_multiclass_LDA'], 'mean_efficiency',...
    'log_efficiency','mean_log_efficiency','trials','-v6');

% convert to table
values = cell(2,1); 
values{2} = features;
T = ndarray2table(log_efficiency, {'Subject' 'Features'}, values);
writetable(T,[resultsdir 'table_WakemanHenson_multiclass_LDA.csv'])
