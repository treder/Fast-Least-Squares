% Hastie et al (1995) investigated the relationship between multi-class
% LDA, CCA, and optimal scoring (a regression approach). Of particular
% importance for the fast training/testing paper is the relationship
% between OS and LDA because it allows to (partly) frame multi-class LDA as
% a regression problem.
%
% Reference:
% Hastie, T., Buja, A., & Tibshirani, R. (1995). Penalized Discriminant 
% Analysis. The Annals of Statistics, 23(1), 73–102. 
% https://doi.org/10.1214/aos/1176324456

clear all
close all
% addpath ~/svn/mt03/fast_lda_training/

nsamples = 600;
nfeatures = 100;
scale = .02; % 1:nfeatures;

% Tolerance wrt rounding errors when comparing two different calculations
% that should lead to the same analytic solution
tol = 10^-10;

nclasses = 3;
[X, clabel, Y, M] = simulate_gaussian_data(nsamples, nfeatures, nclasses, [0.25,0.25,.5], scale,0);


%% Centering

% Centering: If 1, the features in X are centered. If 0, X is augmented by
% a column of 1's instead
% do_center = 0;
do_center = 1;

if do_center
    % center predictors. It is then not needed to augment the data with a
    % column of 1's, so we set Xa = X centered
    X = bsxfun(@minus, X, mean(X));
    Xa = X;
else
    % augment by column of 1's
    Xa = [X, ones(size(X,1),1)];
end

%% Define some quantities

% Number of samples per class
nc = arrayfun(@(c) sum(clabel == c), 1:nclasses);

% Calculate class means and sample mean
mbar = mean(X);            % sample mean
m = zeros(nclasses, nfeatures);       % class means
for c=1:nclasses
    m(c,:) = mean(X(clabel==c,:));
end


% Covariance matrices (see definition in p.80 in Hastie et al, 1995)
E11 = Y'* Y / nsamples;
E22 = (Xa' * Xa) / nsamples;
E12 = Y' * Xa / nsamples;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         MULTICLASS LDA           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% using generalised eigenvalue decomposition (then usually a nearest
% centroid/prototype classifier is applied)

% Between-classes scatter for multi-class
Sb = zeros(nfeatures);
for c=1:nclasses
    Sb = Sb + nc(c) * (m(c,:)-mbar)'*(m(c,:)-mbar);
end

fprintf('Rank of Sb: %d\n', rank(Sb))

% Within-class scatter
Sw = zeros(nfeatures);
for c=1:nclasses
    Sw = Sw + (nc(c)-1) * cov(X(clabel==c,:));
end

% Note that Hastie et al. use the covariance matrix Sw/nsamples, whereas
% here the within-class scatter matrix is used. For this reason, the OS
% results need to be additionally scaled by sqrt{N}

% Scale as covariance like in Hastie et al (p. 81)
% Sw = Sw / nsamples;

% Generalised eigenvalue problem
tic,[W,D] = eig(Sb, Sw, 'vector');toc
% tic,[W,D] = eigs(Sb, Sw, nclasses-1);toc
[D, so] = sort(D,'descend');
W = W(:,so);

D = D(1:nclasses-1);
W = W(:, 1:nclasses-1);

% Columns of W need to be scaled correctly such that it turns Sw into identity
W  = W * diag(1./sqrt(diag(W'*Sw*W)));

fprintf('\n (result should be identity) W''*Sw*W = \n')
disp(W'*Sw*W)

r1 = norm(W'*Sw*W - eye(size(W,2)));
fprintf('W''*Sw*W = I: diff = %1.5f', r1)
if r1 < tol,      fprintf(': correct.\n')
else              fprintf(': failed\n'), end

r1 = norm(X'*X - (Sw + Sb));
fprintf('X''*X = Sw + Sb: diff = %1.5f', r1)
if r1 < tol,      fprintf(': correct.\n')
else              fprintf(': failed\n'), end
% 
% r1 = norm(X'*X / nsamples - (m'*E11*m + E22));
% fprintf('X''*X / N= m*E11*m'' + Sb: diff = %1.5f', r1)
% if r1 < tol,      fprintf(': correct.\n')
% else              fprintf(': failed\n'), end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          OPTIMAL SCORING           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Multivariate regression on indicator matrix
Beta_OS_raw = (Xa' * Xa)\(Xa' * Y);
Yhat = Xa * Beta_OS_raw; 

% Eigenanalysis of Y' Yhat [latter term in ASR]: Scale D_OS by N^-1 to obtain the quantity alpha^2
[Theta_OS,D_OS] = eig(Y' * Yhat / nsamples, E11, 'vector');

% Remove the eigenvector corresponding to the eigenvalue 0 or 1
idx_rm = find( abs(D_OS)<tol | abs(D_OS-1)<tol);
D_OS(idx_rm)       = [];
Theta_OS(:,idx_rm) = [];


% Scale Theta_OS correctly to make it diagonal w.r.t E11
Theta_OS  = Theta_OS * diag(1./sqrt(diag(Theta_OS'*E11*Theta_OS)));

fprintf(' Theta_OS'' * E11 * Theta_OS should be identity:\n')
disp(Theta_OS' * E11 * Theta_OS)

fprintf(' (Y*theta)'' * (Y*theta) should be diagonal:\n')
disp((Y*Theta_OS)'*(Y*Theta_OS))

Beta_OS = Beta_OS_raw * Theta_OS;

fprintf('Does Beta_OS diagonalise E22?\n')
disp(Beta_OS' * E22 * Beta_OS)

if ~do_center
   Beta_OS = Beta_OS(1:end-1,:); 
   Beta_OS_raw = Beta_OS_raw(1:end-1,:); 
end


% Do optimal scoring and multi-class LDA span the same subspace?
r1 = subspace(Beta_OS, W);
fprintf('Subspace angle between Beta_OS and W = %1.5f < tol', r1)
if r1 < tol,      fprintf(': correct (same subspace)\n')
else              fprintf(': failed\n'), end

r1 = subspace(Beta_OS_raw(:,2:nclasses), W);
fprintf('Subspace angle between Beta_OS_Raw and W = %1.5f < tol', r1)
if r1 < tol,      fprintf(': correct (same subspace)\n')
else              fprintf(': failed\n'), end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         CCA            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Theta_CCA, S, Beta_CCA] = svd( E11^-1 * E12 * E22^-1);

[Beta_CCA, Theta_CCA, S] = canoncorr(Xa, Y);
% Note that the function canoncorr produces Beta and Theta such that
% 1) Beta' * cov(X) * Beta = I
% 2) Theta' * cov(Y) * Theta = I
%
% in other words, the coefficients diagonalise the unbiased estimator of
% the covariance matrix. When X or Y are not centered, X'*X and Y'*Y are
% not diagonalised by canoncorr.

% As explained above, the coefficients diagonalise the covariance matrices
Beta_CCA' *  cov(Xa) * Beta_CCA
Theta_CCA' * cov(Y) * Theta_CCA


% Should be  : Theta' * E11 * Theta = I
Theta_CCA' * E11 * Theta_CCA


% Should be (but is not): Beta' * E22 * Beta = I
% Beta_CCA = Beta_CCA(:,1:nclasses);
Beta_CCA' *  E22 * Beta_CCA

%% Do CCA by hand using formula (4) in Magnus Borga's (2001) CCA tutorial
[Theta_CCA, Theta_D] = eig( (E11\E12) * (E22\(E12')) );
[Beta_CCA, Beta_D] = eig( (E22\(E12')) * (E11\E12) );

% Make everything real
Theta_CCA = real(Theta_CCA);
Beta_CCA = real(Beta_CCA);
Theta_D = real(Theta_D);
Beta_D = real(Beta_D);

% Sort eigenvalues
[Beta_D, beta_so] = sort(diag(Beta_D),'descend');
[Theta_D, theta_so] = sort(diag(Theta_D),'descend');

Beta_CCA = Beta_CCA(:, beta_so);
Theta_CCA = Theta_CCA(:, theta_so);

Beta_CCA = Beta_CCA(:, 1:nclasses-1);
Theta_CCA = Theta_CCA(:, 1:nclasses-1);
Beta_D = Beta_D(1:nclasses-1);
Theta_D = Theta_D(1:nclasses-1);

% The coefficients Beta and Theta are already correct in that they
% diagonalise E11 and E22. However, the diagonal is not 1, so we need to
% rescale the weight vectors accordingly

% Should be diagonal
Scale_Theta = diag(Theta_CCA' * E11 * Theta_CCA);
Theta_CCA = Theta_CCA/diag(sqrt(Scale_Theta));

fprintf('Theta_CCA'' * E11 * Theta_CCA:\n')

Theta_CCA' * E11 * Theta_CCA

% Should be: Beta' * E22 * Beta = I
Scale_Beta = diag(Beta_CCA' * E22 * Beta_CCA);
Beta_CCA = Beta_CCA/diag(sqrt(Scale_Beta));

fprintf('Beta_CCA'' * E22 * Beta_CCA:\n')
Beta_CCA' *  E22 * Beta_CCA

%% If centered, remove b
if ~do_center
   Beta_CCA = Beta_CCA(1:end-1,:);
   fprintf('Removed bias b.\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    OS   vs    CCA      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Correlation between CCA and OS weights:\n')
corr(Beta_OS, Beta_CCA)

fprintf('Correlation between LDA and CCA weights:\n')
corr(W, Beta_CCA)

fprintf('Correlation between **LDA and OS** weights:\n')
corr(W, Beta_OS)

%% Formula (21) defines the relationship between W and Beta_OS. 

%%% -- not necessary since the information is contained in D_OS = alpha^2
% Calculate ASR (see formula (5) in Hastie et al.)
% ASR = 1 - diag(Theta_OS' * Y' * Yhat * Theta_OS/ nsamples);

% Second part of ASR is alpha^2 
% alpha_squared = diag(Theta_OS' * Y' * Yhat * Theta_OS/ nsamples);

%%% We only need to scale Beta_OS with alpa * (1-alpha)
Beta_OS_scaled = Beta_OS * diag(1 ./ sqrt(D_OS .* (1-D_OS)));


% Note that Hastie et al. use the covariance matrix Sw/nsamples, whereas
% here the within-class scatter matrix is used. For this reason, the OS
% results need to be additionally scaled by sqrt{N}
Beta_OS_scaled  = Beta_OS_scaled / sqrt(nsamples);


fprintf('Correlation between **LDA and OS** weights:\n')
C = corr(W, Beta_OS_scaled)

if abs( abs(C(1,2)) - 1) < tol
    % need to reorder Beta_OS_scaled so that they are in the same order as
    % in W
    Beta_OS_scaled = [Beta_OS_scaled(:,2), Beta_OS_scaled(:,1)];
end

fprintf('Norm of W directions: '), disp([norm(W(:,1)), norm(W(:,2))])
fprintf('Norm of Beta_OS_scaled directions: '), disp([norm(Beta_OS_scaled(:,1)), norm(Beta_OS_scaled(:,2))])

fprintf('If the scaling is correct, the ratio of the norms should be equal for W and Beta_OS:\n')
fprintf('Ratio |W_1|/|W_2| = %3.5f\n', norm(W(:,1))/ norm(W(:,2)))
fprintf('Ratio |B_OS_1|/|B_OS_2| = %3.5f\n',norm(Beta_OS_scaled(:,1))/norm(Beta_OS_scaled(:,2)))
