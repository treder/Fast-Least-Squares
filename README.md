# Fast-Least-Squares
Fast cross-validation for least-squares methods and multi-class LDA

Link to arXiv.org preprint:  http://arxiv.org/abs/1803.10016


## Main functions

* [`fast_least_squares`](fast_least_squares.m): implements the analytical cross-validation approach for least-squares models and multi-class LDA
* [`standard_lda_traintest`](standard_lda_traintest.m): for comparative purposes, implements standard cross-validation for LDA and multi-class LDA

## Simulations

The simulations reported in the paper can be reproduced using the scripts in the subfolder [`simulation`](simulation). The function [`simulate_gaussian_data`](simulation/simulate_gaussian_data.m) creates the multivariate Gaussian data. It is called from the scripts [`simulation1_binary_LDA`](simulation/simulation1_binary_LDA.m) and [`simulation2_multiclass_LDA`](simulation/simulation2_multiclass_LDA.m) which perform the cross-validation and permutations experiments.

## MEEG data

The analyses performed on the [Wakeman and Henson](https://www.nature.com/articles/sdata20151) can be reproduced using the scripts in the subfolder [WakemanHensonData](WakemanHensonData). The data can be downloaded from the [OpenfMRI](https://openfmri.org/dataset/ds000117/) website. 

* [`WakemanHenson_preprocess`](WakemanHensonData/WakemanHenson_preprocess.m): preprocesses the EEG/MEG data using Fieldtrip. 
* [`WakemanHenson_run_permutations_binary_LDA`](WakemanHensonData/WakemanHenson_run_permutations_binary_LDA.m): performs the permutations analysis for binary LDA
* [`WakemanHenson_run_permutations_multiclass_LDA`](WakemanHensonData/WakemanHenson_run_permutations_multiclass_LDA.m): performs the permutations analysis for multi-class LDA
* [`results_WakemanHenson`](WakemanHensonData/results_WakemanHenson.m): collects the results of the permutation analyses and stores it in tables
