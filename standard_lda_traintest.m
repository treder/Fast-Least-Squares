function [perf, cfg] = standard_lda_traintest(cfg, X, y)
% Standard way to do CROSS VALIDATION and PERMUTATIONS wherein a classifier 
% is trained from sketch in every iteration. The function is called very
% similarly to fast_least_squares.
%
% Usage:
% perf = standard_lda_traintest(cfg, X, y)
%
%Parameters:
% X              - [samples x features] data matrix
% y              - [samples x 1] vector of class labels
%
% cfg is a struct with fields:
% .metric       - classifier performance metric, default 'acc'. See
%                 mv_classifier_performance. If set to [] or 'none', the
%                 raw classifier output (labels or dvals depending on
%                 cfg.cf_output) for each sample is returned. Note that for
%                 multiclass LDA, only 'acc' is supported
% .reg          - type of regularisation, 'none' for no regularisation.
%                 'ridge': ridge-type regularisation of C + lambda*I,
%                          where C is the covariance matrix and I is the
%                          identity matrix
%                 'shrink': shrinkage regularisation using (1-lambda)*C +
%                          lambda*nu*I, where nu = trace(C)/F and F =
%                          number of features. nu assures that the trace of
%                          C is equal to the trace of the regularisation
%                          term. Note that shrinkage precludes low-rank
%                          updates. Therefore, the shrinkage lambda is
%                          translated into the corresponding ridge
%                          parameter to approximate shrinkage with the
%                          ridge penalty.
%                  (default 'shrink')
% .lambda        - value of the lambda parameter. If
%                  regularisation='shrink' and lambda='auto', the
%                  Ledoit-Wolf automatic estimation procedure is used to
%                  estimate lambda in each iteration. (default 'auto')
%
% CROSS-VALIDATION fields:
% .cv           - perform cross-validation, can be set to 'kfold',
%                 'leaveout', 'holdout', or 'none' (default 'kfold')
% .k            - number of folds in k-fold cross-validation (default 5)
% .stratify     - if 1, the class proportions are approximately preserved
%                 in each fold (default 1)
% .frac         - if cv is 'holdout', frac is the fraction of test samples
%                 (default 0.1)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds (default 1)
%
% PERMUTATION fields:
% .nperm        - number of permutations to perform. In each permutation,
%                 the labels are randomly shuffled. If 0, no permutations
%                 are performed (default 1000)
%
%Output:
% perf - [(nperm+1) x 1] vector with performance measures for all
%        permutations. Note that perf(1) corresponds to the 
%        original (unpermuted) class labels. The results of the
%        permutations are appended as the following positions (
%        perf(2:end))
% cfg  - configuration struct. Same as input cfg, but missing fields have
%        been filled with default values

% (c) Matthias Treder 2018

[N,P] = size(X);
y = y(:);

nclasses = numel(unique(y));

mv_set_default(cfg,'metric','acc');
mv_set_default(cfg,'fast',1);
mv_set_default(cfg,'feedback',1);
mv_set_default(cfg,'nperm',1);

% Regularisation
mv_set_default(cfg,'reg','ridge'); % 'shrink'
mv_set_default(cfg,'lambda',1); % 'auto'

% Cross-validation settings
mv_set_default(cfg,'cv','kfold');
mv_set_default(cfg,'repeat',2);
mv_set_default(cfg,'k',10);
mv_set_default(cfg,'frac',0.1);
mv_set_default(cfg,'stratify',1);

if strcmp(cfg.metric,'auc') && nclasses > 2
    error('For multi-class LDA, only ''acc'' is supported as performance metric')
end

%% Regularisation term
if strcmp(cfg.reg,'shrink')
    % Convert shrinkage regularisation term into ridge regularisation term.
    % This is not actually necessary for classical LDA, but it is done here
    % to assure comparability with the fast LDA code
    lambda = cfg.lambda/(1-cfg.lambda) * trace(X'*X)/P;
    cfg.reg = 'ridge';
else
    lambda = cfg.lambda;
end

%% Set up classifier parameters for call to MVPA-Light
if nclasses == 2
    param = mv_get_classifier_param('lda');
    param.reg       = cfg.reg;
    param.lambda    = lambda;
    output = 'dval';
    trainfun = @train_lda;
    testfun = @test_lda;
else
    param = mv_get_classifier_param('multiclass_lda');
    param.reg       = cfg.reg;
    param.lambda    = lambda;
    output = 'clabel';
    trainfun = @train_multiclass_lda;
    testfun = @test_multiclass_lda;
end

%% Run permutations and cross-validation

% Average classification performance across repeats and test folds
avdim = [1,2];

% Classifier performance metric
perf = zeros(cfg.nperm+1, 1);

% Predicted decision values obtained with updating
% ypred = zeros(N,1);

% Stores the classifier outputs for one cross-validation (with possibly
% multiple repeats)
cf_output = cell(cfg.repeat, cfg.k);
testlabel = cell(cfg.repeat, cfg.k);

for pp=1:cfg.nperm+1        % ---- Permutations ----
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%   BINARY LDA and MULTI-CLASS LDA     %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Binary LDA and multi-class LDA only need different train/test
    % functions defined above, the rest of the code is identical
    
    % Randomly shuffle class labels (except for permutation #1 which
    % corresponds to the original data)
    if pp > 1
        y = y(randperm(N));
    end
    
    if ~strcmp(cfg.cv,'none')

        for rr=1:cfg.repeat                 % ---- CV repetitions ----
            
            cv = mv_get_crossvalidation_folds(cfg.cv, y, cfg.k, cfg.stratify, cfg.frac);
            
            for kk=1:cv.NumTestSets                     % ---- CV folds ----
                
                % Get train data
                Xtrain = X(cv.training(kk),:);
                
                % Get train and test labels
                trainlabel= y(cv.training(kk));
                testlabel{rr,kk} = y(cv.test(kk));

                % Train classifier on training data
                cf= trainfun(param, Xtrain, trainlabel);
                
                % Obtain classifier output (labels or dvals) on test data
                cf_output{rr,kk} = mv_get_classifier_output(output, cf, testfun, X(cv.test(kk),:));

            end
        end

        % Calculate performance for current permutation
        perf(pp) = mv_calculate_performance(cfg.metric, cf_output, testlabel, avdim);

    else
        % No cross-validation, just train and test once for each
        % training/testing time. This gives the classification performance for
        % the training set, but it may lead to overfitting and thus to an
        % artifically inflated performance.

        % We already calculated the decision values per yhat = H * y so
        % we only need to get the clasifier performance metric

        % We need to flip the sign because
        % in mv_calculate_performance, it is assumed that class 1
        % has larger decision values than class 2, but in the
        % regression approach class 2 has larger values than class
        % 1
        error('todo')
        perf(pp) = mv_calculate_performance(cfg.metric, -yhat, y, avdim);

        testlabel = y;
        avdim = [];
    end
end

%% Output arguments


stat = [];
stat.perf = perf;
