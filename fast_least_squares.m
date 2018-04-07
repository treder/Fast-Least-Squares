function [perf, cfg] = fast_least_squares(cfg, X, y)
% Fast CROSS_VALIDATION and PERMUTATIONS for regularised binary or
% multi-class Linear Discriminant Analysis (LDA). For details on the 
% algorithm, see Treder (2018).
%
% Note: Uses MVPA-Light functions (github.com/treder/MVPA-Light).
%
% Usage:
% stat = fast_lda_permutations(cfg, X, y)
%
%Parameters:
% X              - [samples x features] data matrix
% y              - [samples x 1] vector. For classification problems (LDA
%                  and multi-class LDA), the vector should contain the
%                  class labels, i.e. 1's (for class 1), 2's (for class 2),
%                  3's and so on.
%                  For linear regression or ridge regression, y is the
%                  response variable.
%
% cfg is a struct with fields:
% .multiclass   - if 0, cross-validation is performed for least-squares
%                 models (linear regression, ridge regression, LDA). If 1, 
%                 multi-class LDA is performed (default 0)
% .metric       - classifier performance metric, default 'acc'. See
%                 mv_classifier_performance. If set to [] or 'none', the
%                 raw classifier output (labels or dvals depending on
%                 cfg.cf_output) for each sample is returned.
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
% .CV           - perform cross-validation, can be set to 'kfold',
%                 'leaveout', 'holdout', or 'none' (default 'kfold')
% .k            - number of folds in k-fold cross-validation (default 5)
% .stratify     - if 1, the class proportions are approximately preserved
%                 in each fold (default 1)
% .frac         - if CV is 'holdout', frac is the fraction of test samples
%                 (default 0.1)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds (default 1)
%
% PERMUTATION fields:
% .nperm        - number of permutations to perform. In each permutation,
%                 the labels are randomly shuffled and then
%                 cross-validation is performed according to the
%                 cross-validation settings above.
%                 If 0, no permutations are performed, and cross-validation
%                 is performed once using the original/unshuffled data
%                 (default 1000)
%
%Output:
% perf - [(nperm+1) x 1] vector with performance measures for all
%        permutations. Note that perf(1) corresponds to the
%        original (unpermuted) class labels. The results of the
%        permutations are appended as the following positions (
%        perf(2:end))
% cfg  - configuration struct. Same as input cfg, but missing fields have
%        been filled with default values
%

% (c) Matthias Treder 2018

[N,P] = size(X);
y = y(:);

nclasses = numel(unique(y));

% Tolerance wrt rounding errors
tol = 10^-10;

mv_set_default(cfg,'multiclass',0);
mv_set_default(cfg,'metric','acc');
mv_set_default(cfg,'feedback',1);
mv_set_default(cfg,'nperm',1);

% Regularisation
mv_set_default(cfg,'reg','ridge'); % 'shrink'
mv_set_default(cfg,'lambda',1); % 'auto'

% Cross-validation settings
mv_set_default(cfg,'CV','kfold');
mv_set_default(cfg,'repeat',2);
mv_set_default(cfg,'k',10);
mv_set_default(cfg,'frac',0.1);
mv_set_default(cfg,'stratify',1);

%% Regularisation term
I0 = eye(P+1);
I0(end) = 0;        % don't regularise bias term

if strcmp(cfg.reg,'shrink')
    % Convert shrinkage regularisation term into ridge regularisation term
    lambda = cfg.lambda/(1-cfg.lambda) * trace(X'*X)/P;
else
    lambda = cfg.lambda;
end

%% Augment data with column of 1's (for bias/intercept)
X = [X, ones(size(X,1),1)];

%% Hat matrix
H = X * ( (X'*X + lambda * I0) \ X');

% I - H (identity minus hat matrix) is the one we need to invert for updates
IH = eye(N) - H;

%% Run permutations and cross-validation

% Average classification performance across repeats and test folds
avdim = [1,2];

% Classifier performance metric
perf = zeros(cfg.nperm+1,1);

% Stores the classifier outputs for one cross-validation (with possibly
% multiple repeats)
cf_output = cell(cfg.repeat, cfg.k);
testlabel = cell(cfg.repeat, cfg.k);

if ~cfg.multiclass
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% LEAST-SQUARES (LINEAR/RIDGE REGRESSION, BINARY LDA)  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for pp=1:cfg.nperm+1        % ---- Permutations ----
        
        % Randomly shuffle class labels (except for permutation #1 which
        % corresponds to the original data)
        if pp > 1
            y = y(randperm(N));
        end
        
        % Produce fitted values using the full model
        yhat = H * y;
        
        % for debugging
        % plot(y), hold all, plot(yhat)
        
        % Calculate error
        ehat = y - yhat;
        
        if ~strcmp(cfg.CV,'none')
            
            for rr=1:cfg.repeat                 % ---- CV repetitions ----
                
                CV = mv_get_crossvalidation_folds(cfg.CV, y, cfg.k, cfg.stratify, cfg.frac);
                
                for kk=1:CV.NumTestSets                     % ---- CV folds ----
                    
                    % Logical indices of test samples
                    Te = CV.test(kk);
                    
                    % Cross-validated decision values on test set
                    einv = IH(Te,Te) \ ehat(Te);
                    ycv_Te = y(Te) - einv;
                    
                    % Need to fix b if metric = 'acc' (always need to do so
                    % since the classes are not coded as +1 and -1)
                    if strcmp(cfg.metric,'acc')
                        Tr = CV.training(kk);
                        % Calculate cross-validated decision  values on 
                        % training set using the update formula on the 
                        % training set
                        ycv_Tr = y(Tr) - ( ehat(Tr) + H(Tr,Te) * einv );
                        
                        % By calculating the projected means (rather than
                        % calculating the means and then projecting) we avoid
                        % the need to explicitly calculate the weight vector
                        mbar = mean(ycv_Tr);
                        m1 = mean(ycv_Tr(y(Tr)==1));
                        m2 = mean(ycv_Tr(y(Tr)==2));
                        
                        % Calculate the correction for bias b [see formula 
                        % in Appendix B]
                        N1 = sum(y(Tr)==1);
                        N2 = sum(y(Tr)==2);
                        b_correct = - (m1+m2)/2 - ( (N1 + N2*2)/CV.TrainSize(kk) - mbar );
                        
                        % Perform correction
                        ycv_Te = ycv_Te + b_correct;
                    end
                    
                    % Store testlabels
                    testlabel{rr,kk} = y(CV.test(kk));
                    
                    % Store decision values. We need to flip the sign because
                    % in mv_calculate_performance, it is assumed that class 1
                    % has larger decision values than class 2, but in the
                    % regression approach class 2 has larger values than class
                    % 1
                    cf_output{rr,kk} = -ycv_Te;
                    
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
            perf(pp) = mv_calculate_performance(cfg.metric, -yhat, y, avdim);
            
            testlabel = y;
            avdim = [];
        end
    end
    
elseif nclasses > 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%            MULTI-CLASS LDA           %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Create class indicator matrix from class labels
    Y = zeros(N, nclasses);
    for cc=1:nclasses
        Y(y==cc, cc) = 1;
    end
    
    for pp=1:cfg.nperm+1        % ---- Permutations ----

        % Randomly shuffle rows of indicator matrix (except for 
        % permutation #1 which corresponds to the original data)
        if pp > 1
            ridx = randperm(N);
            Y = Y(ridx, :);
            y = y(ridx, :);   % original class label vector is needed for CV definition
        end
        
        % Produce fitted values using the full model
        Yhat = H * Y;
        
        % Calculate error
        Ehat = Y - Yhat;
        
        if ~strcmp(cfg.CV,'none')
            
            for rr=1:cfg.repeat                 % ---- CV repetitions ----
                
                CV = mv_get_crossvalidation_folds(cfg.CV, y, cfg.k, cfg.stratify, cfg.frac);
                
                for kk=1:CV.NumTestSets                     % ---- CV folds ----

                    % Logical indices of train and test samples
                    Tr = CV.training(kk);
                    Te = CV.test(kk);
                    
                    %%% --- Step 1: updating Y ---
                    % Here, the update rule is applied twice to obtain
                    % cross-validated decision values on test set and
                    % training set
                    
                    % Cross-validated decision values on test set
                    Einv = IH(Te,Te) \ Ehat(Te,:);
                    Ycv_Te = Y(Te,:) - Einv;
                    
                    % Calculate decision values on training set using the
                    % updating formula on the training set
                    Ycv_Tr = Y(Tr,:) - (Ehat(Tr,:) + H(Tr,Te) * Einv);
                    
                    %%% --- Step 2: eigenanalysis ---
                    % Here, eigenanalysis of Ycv_Tr '* Y(Tr,:) is performed
                    % to obtain THETA and the scaling matrix D
                    
                    % Required metric [diagonal matrix with class
                    % frequencies]
                    E11 = diag(sum(Y(Tr,:))) / CV.TrainSize(kk);
                    
                    % Eigenanalysis of Y' Yhat [latter term in ASR]: Scale D_OS by N^-1 to obtain the quantity alpha^2
                    [THETA,D_OS] = eig(Ycv_Tr' * Y(Tr,:) / CV.TrainSize(kk), E11, 'vector');
                    
                    % Remove the trivial eigenvector corresponding to the 
                    % eigenvalue 0 or 1
                    idx_rm = find( abs(D_OS)<tol | abs(D_OS-1)<tol);
                    D_OS(idx_rm)       = [];
                    THETA(:,idx_rm) = [];
                    
                    % Scale THETA correctly to make it diagonal wrt E11
                    THETA  = THETA * diag(1./sqrt(diag(THETA'*E11*THETA)));

                    % Scaling matrix D
                    D = diag(1 ./ sqrt(D_OS .* (1-D_OS))) / sqrt(CV.TrainSize(kk));

                    % Desired training and test scores in discriminant space
                    y_LDA_Tr = Ycv_Tr * THETA * D;
                    y_LDA_Te = Ycv_Te * THETA * D;
                    
                    % Training scores are used to calculate the class
                    % centroids in discriminant space
                    centroid = zeros(nclasses, nclasses-1);
                    for c=1:nclasses
                        centroid(c,:) = mean(y_LDA_Tr( y(CV.training(kk))==c ,:) );
                    end
                    
                    % To calculate predicted labels, first calculate 
                    % Euclidean distance of each sample to each class
                    % centroid
                    dist = arrayfun( @(c) sum( (bsxfun(@minus, y_LDA_Te, centroid(c,:))).^2, 2), 1:nclasses, 'Un',0);
                    dist = cat(2, dist{:});
                    
                    % For each sample, find the closest class centroid
                    predlabel = zeros(CV.TestSize(kk),1);
                    for ii=1:CV.TestSize(kk)
                        [~, idx] = min(dist(ii,:));
                        predlabel(ii) = idx;
                    end

                    % Store testlabels
                    testlabel{rr,kk} = y(CV.test(kk));
                    
                    % Store predicted labels
                    cf_output{rr,kk} = predlabel;
                    
                end
            end
            
            % Calculate performance for current permutation
            perf(pp) = mv_calculate_performance(cfg.metric, cf_output, testlabel, avdim);
            
        end
    end
   

end
%% Output arguments

stat = [];
stat.perf = perf;
