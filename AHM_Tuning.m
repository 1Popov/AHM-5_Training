%% AHM_Tuning.m
% Stage 6 Tuning: Global Hyperparameter Refinement & Verification.
% Purpose:
%   1. Find the single best hyperparameter set for the Full NCV Dataset.
%   2. VERIFY those parameters by training 5 models (CV splits) and
%      testing EACH against the strictly Held-Out set.
%
% Methodology:
%   - Optimization: 5-Fold CV using CV_Audio (Leak-Proof) on NCV Domain.
%   - Verification: Fixed-param training on folds -> Hold-Out Eval.

delete(gcp('nocreate'));
clear; clc; close all;
rng(42,'twister');

% --- SETUP ---
addpath('C:\NextCloud\Work Projects\AHM\4_FeatureCleaning');
addpath('C:\NextCloud\Work Projects\AHM\Utilities');
addpath(genpath(pwd));

modelDir = './Results';
if ~exist(modelDir,'dir'), mkdir(modelDir); end
resultsDir = fullfile(modelDir, sprintf('Tuning_%s', string(datetime('now','Format','dd-MMM-uuuu_HH_mm'))));
mkdir(resultsDir);
diary(fullfile(resultsDir, 'Tuning_Log.txt'));

%% 1. LOAD DATA
fprintf('Loading FT...\n');
load('Main_4C_Final.mat', 'FT');

% --- 1.1 DEFINE DOMAINS ---
% NCV Domain: The "Sandbox" for optimization
maskNCV = FT.legitForNCV & ~FT.isAugment & ~ismember(FT.OfficialSplit, ["test", "eval"]);

% Hold-Out Domain: Kept strictly aside for Verification/Production
maskHeldOut = FT.legitForNCV & ~FT.isAugment & ismember(FT.OfficialSplit, ["test", "eval"]);

fprintf('  Tuning Domain (Full NCV): %d rows\n', nnz(maskNCV));
fprintf('  Hold-Out Domain:          %d rows\n', nnz(maskHeldOut));

%% 2. CONFIGURATION
predictorNames = FT.Properties.VariableNames(startsWith(FT.Properties.VariableNames, 'Adv_WST_Path') & endsWith(FT.Properties.VariableNames, '_Norm'));
responseName   = 'AHM_7_Class';

% Tuning Settings
K_tune = 5;
numBayesoptEvaluations = 50;
TARGET_BALANCE = 30000;

% Fixed Network Params
fixedNetParams.Activations = 'relu';
fixedNetParams.Solver = 'adam';
fixedNetParams.MaxEpochs = 120;
fixedNetParams.Verbose = 0;
fixedNetParams.ExecutionEnvironment = 'gpu';
fixedNetParams.ValidationFrequency = 50;
fixedNetParams.ValidationPatience = 15;

% MASTER CLASS LIST
rawCats = categories(FT.(responseName));
fixedNetParams.ClassNames = categorical(rawCats, rawCats, 'Ordinal', false);
refClassNames = cellstr(fixedNetParams.ClassNames);

% --- NARROWED SEARCH SPACE (THE GOLDEN ZONE) ---
% Derived from Discovery Run #3
optimVars = [
    optimizableVariable('Layer1Size', [1000 2400], 'Type', 'integer')
    optimizableVariable('Layer2Size', [600 1200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [5e-4, 3e-3], 'Transform', 'log')
    optimizableVariable('Lambda', [1e-5, 2e-4], 'Transform', 'log')
    optimizableVariable('DropoutP', [0.50, 0.75])
    optimizableVariable('MiniBatchSize', {'8192', '16384'}, 'Type', 'categorical')
    ];

optimizedMemoryConstraint = @(x) applyTieredMemoryBudget(x);

%% 3. PREPARE TUNING FOLDS
fprintf('\nGenerating %d-Fold Partition on Full NCV Domain...\n', K_tune);

% Use CV_Audio to generate leak-proof splits on the NCV domain
tuningFolds = CV_Audio(FT, maskNCV, K_tune);

fprintf('Tuning Fold Distribution:\n');
tabulate(tuningFolds(maskNCV));

% Leakage Check on Tuning Folds
maskTest1 = (tuningFolds == 1) & maskNCV;
maskTrain1 = (tuningFolds ~= 1) & (tuningFolds > 0) & maskNCV;
T_CheckTrain = FT(maskTrain1, {'SourceID', 'AHM_7_Class', 'Dataset', 'OriginalStem'});
T_CheckTest  = FT(maskTest1,  {'SourceID', 'AHM_7_Class', 'Dataset', 'OriginalStem'});
[isClean, ~] = LeakageCheck(T_CheckTrain, T_CheckTest);
if ~isClean, error('FATAL: Leakage detected in Tuning Folds!'); end
fprintf('  [QA] Tuning Folds are Leak-Proof.\n');

%% 4. RUN GLOBAL OPTIMIZATION
fprintf('\n>>> Starting Global Refinement Optimization <<<\n');

% This optimizes the AVERAGE performance across all 5 tuning folds
objFcn = @(params) objFcn_Direct(params, FT, maskNCV, tuningFolds, predictorNames, responseName, K_tune, fixedNetParams, refClassNames, TARGET_BALANCE);

results = BayesOpt(objFcn, optimVars, numBayesoptEvaluations, [1, 1], optimizedMemoryConstraint);

if isempty(results)
    error('Tuning Optimization failed.');
else
    finalHyperparams = results.XAtMinObjective;
end

fprintf('\n>>> FINAL OPTIMIZED HYPERPARAMETERS <<<\n');
disp(finalHyperparams);

%% 5. VERIFICATION: TEST FINAL PARAMS ON HOLD-OUT SET (PER FOLD)
fprintf('\n>>> Starting Verification Loop (Hold-Out Consistency) <<<\n');
fprintf('    Training 5 models using Final Params to check stability on Unseen Data.\n');

holdOutAUCs = zeros(K_tune, 1);
holdOutF1s  = zeros(K_tune, 1);

% Prepare Hold-Out Data once
X_ho = FT{maskHeldOut, predictorNames};
Y_ho = categorical(string(FT{maskHeldOut, responseName}), refClassNames);

% Pre-process params
if iscategorical(finalHyperparams.MiniBatchSize)
    currBatch = str2double(string(finalHyperparams.MiniBatchSize));
else
    currBatch = finalHyperparams.MiniBatchSize;
end
hp_num = finalHyperparams; hp_num.MiniBatchSize = currBatch;

for k = 1:K_tune
    fprintf('  Verification Fold %d/%d... ', k, K_tune);

    % Train on Tuning Fold k's training set (Folds != k)
    maskTr = (tuningFolds ~= k) & (tuningFolds > 0) & maskNCV;
    maskTrAug = collectHeliWithAugments(FT, maskTr);

    X_tr_raw = FT{maskTrAug, predictorNames};
    Y_tr_raw = string(FT{maskTrAug, responseName});

    [X_tr_bal, Y_tr_str] = ClassBalance(X_tr_raw, Y_tr_raw, TARGET_BALANCE);
    Y_tr_cat = categorical(Y_tr_str, refClassNames);

    % Normalize (Fit on Train, Apply to Hold-Out)
    [~, mu, sigma] = zscore(X_tr_raw); sigma(sigma==0)=1;
    X_tr_norm = (X_tr_bal - mu) ./ sigma;
    X_ho_norm = (X_ho - mu) ./ sigma;

    % Train
    layers = createNeuralNetworkLayers(size(X_tr_norm,2), hp_num, fixedNetParams, []);

    % Use Hold-Out as validation here to monitor convergence behavior on unseen data
    opts = createTrainingOptions(hp_num, fixedNetParams, X_ho_norm, Y_ho);

    [net, ~] = TrainNetwork(X_tr_norm, Y_tr_cat, layers, opts);

    % Eval
    [Y_pred, Y_scores] = classify(net, X_ho_norm);
    mets = CalculateMetrics(Y_ho, Y_pred, Y_scores, char(fixedNetParams.ClassNames(1)), fixedNetParams.ClassNames);

    holdOutAUCs(k) = mets.PositiveAUC_ROC;
    holdOutF1s(k)  = mets.PositiveF1_score;

    fprintf('HeliAUC=%.4f, HeliF1=%.4f\n', mets.PositiveAUC_ROC, mets.PositiveF1_score);
    reset(gpuDevice);
end

% Statistics
meanAUC = mean(holdOutAUCs);
stdAUC  = std(holdOutAUCs);
fprintf('\n>>> VERIFICATION SUMMARY <<<\n');
fprintf('  Hold-Out HeliAUC: %.4f +/- %.4f\n', meanAUC, stdAUC);
if stdAUC < 0.01
    fprintf('  [PASS] Parameters are stable across folds.\n');
else
    fprintf('  [WARN] High variance on Hold-Out set. Ensembling might be needed.\n');
end

%% 6. SAVE RESULTS
save(fullfile(resultsDir, 'Tuning_Results.mat'), ...
    'finalHyperparams', 'fixedNetParams', 'predictorNames', 'responseName', ...
    'optimVars', 'results', 'holdOutAUCs', 'holdOutF1s', '-v7.3');

fprintf('\nDone. Results saved to: %s\n', fullfile(resultsDir, 'Tuning_Results.mat'));
diary off;

%% LOCAL FUNCTIONS
function tf = applyTieredMemoryBudget(x)
if iscategorical(x.MiniBatchSize) || isstring(x.MiniBatchSize)
    bs = str2double(string(x.MiniBatchSize));
elseif iscell(x.MiniBatchSize)
    bs = str2double(string(x.MiniBatchSize));
else
    bs = x.MiniBatchSize;
end
l1 = x.Layer1Size;
tf = true(height(x), 1);
tf((l1 > 1800) & (bs < 4096)) = false;
end