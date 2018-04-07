% Preprocesses the Wakeman & Henson (2015) dataset. 
%
% Fieldtrip is required to run the code. The raw fiff data is imported into
% Matlab, epochs corresponding to faces and scrambled faces are created.
% 
% Link to dataset:
% https://openfmri.org/dataset/ds000117/
%
% Reference:
% Wakeman, D. G., & Henson, R. N. (2015). A multi-subject, multi-modal 
% human neuroimaging dataset. Scientific Data, 2, 150001. 
% https://doi.org/10.1038/sdata.2015.1

clear
close all

% Path to folder where the raw data has been unpacked. MEG data for subject
% 1 needs to be in the subfolder sub001/MEG, and likewise for the other
% subjects.
loaddir = 'D:/data/WakemanHenson/raw/ds117/';

% Path to folder where the preprocessed data should be stored
savedir  = 'D:/data/WakemanHenson/preproc/';

% Experiment is split into 6 different runs
nRuns  = 6;

% Not all subject numbers exist
sbjs = [1:9, 12:16, 18:19];

for ss=13:numel(sbjs)
     
    sbj = sbjs(ss);
    
    fprintf('\n\n*** Processing subject %d ***\n', sbj)
    filestart  = [loaddir 'sub' sprintf('%03d',sbj) '/MEG/'];
    
    %% Build the filenames: we need to include the number of the run
    filenames = cell(1,nRuns);
    
    for ii=1:nRuns
        filenames{ii} = sprintf([filestart 'run_0%d_sss.fif'], ii);
    end
    
    %% Read header and events for each run
    hdr = cell(1,nRuns);
    ev = cell(1,nRuns);
    
    for ii=1:nRuns
        fprintf('\n--- Run %d ---\n',ii)
        
        %%% --- Read header ---
        hdr{ii} = ft_read_header(filenames{ii});
       
        %%% --- Read trigger events ---
        % (we actually do not necessarily need this but just in case ...)
        cfg = [];
        cfg.dataset             = filenames{ii};
        cfg.trialdef.eventtype = '?';
        ev{ii}                 = ft_definetrial(cfg);
        
    end
    
    %% Trial definitions for epoching the data
    
    % From the README file:
    % Trigger codes 5, 6, 7, 13, 14, 15 code for faces,
    % while codes 17, 18, 19 code for scrambled images.
    trialdef_Faces = cell(1,nRuns);
    trialdef_Scrambled = cell(1,nRuns);
    
    cfg                         = [];
    cfg.trialfun                = 'ft_trialfun_general'; % this is the default
    cfg.trialdef.eventtype      = 'STI101';
    cfg.trialdef.prestim        = 0.5; % in seconds
    cfg.trialdef.poststim       = 1; % in seconds
    
    for ii=1:nRuns
        fprintf('\n--- Run %d ---\n',ii)
        cfg.dataset                 = filenames{ii};
        
        %%% --- Faces ---
        cfg.trialdef.eventvalue     = [5,6,7,13,14,15];
        trialdef_Faces{ii} = ft_definetrial(cfg);
        
        %%% --- Scrambled ---
        cfg.trialdef.eventvalue     = [17,18,19];
        trialdef_Scrambled{ii} = ft_definetrial(cfg);
        
    end
    
    %% Load, epoch and preprocess data
    %%% Select only EEG channels, lowpass filter, baseline correction
    faces = cell(1,nRuns);
    scrambled = cell(1,nRuns);
    
    channel = '*EG*';   % load EEG and MEG channels
    
    for ii=1:nRuns
        fprintf('\n--- Run %d ---\n',ii)
        
        % Get FACES condition
        cfg = trialdef_Faces{ii};
        
        cfg.channel         = channel;  
        cfg.demean          = 'yes';
        cfg.baselinewindow  = [-0.5, 0];
        
        cfg.dataset                 = filenames{ii};
        
        faces{ii}= ft_preprocessing(cfg);
        
        % Get SCRAMBLED condition
        cfg = trialdef_Scrambled{ii};
        
        cfg.channel         = channel;
        cfg.demean          = 'yes';
        cfg.baselinewindow  = [-0.5, 0];
        
        cfg.dataset                 = filenames{ii};
        
        scrambled{ii}= ft_preprocessing(cfg);
    end
    
    % Append data from faces and scrambled conditions
    faces = ft_appenddata([], faces{:});
    scrambled = ft_appenddata([], scrambled{:});
    
    % Append all data together
    % The .trialinfo contains the original trigger codes so we have information
    % which class is 'faces' and which is 'scrambled'
    dat = ft_appenddata([], faces, scrambled);
    
    %% Resample
    cfg = [];
    cfg.resamplefs  = 200;
    dat = ft_resampledata(cfg,dat);
    
    %% Save data
    fprintf('Saving subject %d\n\n', sbj)
    save([savedir sprintf('sub%03d',sbj)],'dat','ev','hdr');
    
end

fprintf('Finished all.\n')