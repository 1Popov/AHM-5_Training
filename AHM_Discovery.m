%% AHM_Discovery.m
% Stage 5 Discovery: Hyperparameter Tuning (7-Class) via Nested CV.
% STRICT CONFIGURATION: 5x5 Folds + SourceID Leakage Checks + Direct WST.
% Includes Vectorized Memory/Compute Constraints to fix BayesOpt crash.

delete(gcp('nocreate'));
clear; clc; close all;
rng(42,'twister');

% --- SETUP ---
addpath('C:\NextCloud\Work Projects\AHM\4_FeatureCleaning');
addpath('C:\NextCloud\Work Projects\AHM\Utilities');
addpath(genpath(pwd));

modelDir = './Results';
if ~exist(modelDir,'dir'), mkdir(modelDir); end
resultsDir = fullfile(modelDir, sprintf('Run_%s', string(datetime('now','Format','dd-MMM-uuuu_HH_mm'))));
mkdir(resultsDir);
diary(fullfile(resultsDir, 'Discovery.txt'));

%% 1. LOAD DATA
fprintf('Loading FT...\n');
load('Main_4C_Final.mat', 'FT');

% --- 1.1 DEFINE NCV DOMAIN ---
maskNCV = FT.legitForNCV & ~FT.isAugment;

% --- 1.2 ASSIGN OUTER FOLDS (In-Memory) ---
fprintf('Assigning Outer Folds...\n');
FT.FoldOuter = zeros(height(FT), 1);

% A. Respect ESC50 Official Folds (Standard Benchmark)
maskESC = maskNCV & (FT.Dataset == "ESC50");
if any(maskESC)
    esc_splits = string(FT.OfficialSplit(maskESC));
    esc_folds = double(erase(esc_splits, "fold"));
    esc_folds(isnan(esc_folds)) = 0;
    FT.FoldOuter(maskESC) = esc_folds;
    fprintf('  Mapped %d ESC50 rows to official folds.\n', nnz(esc_folds > 0));
end

% B. Define Held-Out Sets (Strictly Excluded from NCV)
maskHeldOut = maskNCV & ismember(FT.OfficialSplit, ["test", "eval"]);
fprintf('  Held-Out Rows (Test/Eval): %d (FoldOuter = 0)\n', nnz(maskHeldOut));

% C. Assign Folds for Rest (CV_Audio)
maskRest = maskNCV & (FT.FoldOuter == 0) & ~maskHeldOut;

if any(maskRest)
    fprintf('  Running CV_Audio on %d remaining rows (FSD50K-Dev, MAD-Train, Others)...\n', nnz(maskRest));
    folds_full = CV_Audio(FT, maskRest, 5);
    FT.FoldOuter(maskRest) = folds_full(maskRest);
end

fprintf('Outer Fold Distribution:\n');
tabulate(FT.FoldOuter(maskNCV & FT.FoldOuter > 0));

%% 2. CONFIGURATION
% Select only the 90 Normalized WST features
predictorNames = FT.Properties.VariableNames(startsWith(FT.Properties.VariableNames, 'Adv_WST_Path') & endsWith(FT.Properties.VariableNames, '_Norm'));
responseName   = 'AHM_7_Class';

K_outer = 5;
K_inner = 5;
numBayesoptEvaluations = 60;
TARGET_BALANCE = 30000; % Global Consistency Fix

% Fixed Network Params
fixedNetParams.Activations = 'relu';
fixedNetParams.Solver = 'adam';
fixedNetParams.MaxEpochs = 100;
fixedNetParams.Verbose = 0;
fixedNetParams.ExecutionEnvironment = 'gpu';
fixedNetParams.ValidationFrequency = 50;
fixedNetParams.ValidationPatience = 12;

% MASTER CLASS LIST - Enforce Strict Categorical Order
rawCats = categories(FT.(responseName));
fixedNetParams.ClassNames = categorical(rawCats, rawCats, 'Ordinal', false);
refClassNames = cellstr(fixedNetParams.ClassNames);

% --- OPTIMIZED SEARCH SPACE (SHIFTED RANGES) ---
% Adjusted for Raw WST features (higher capacity, higher regularization)
optimVars = [
    optimizableVariable('Layer1Size', [400 1200], 'Type', 'integer')
    optimizableVariable('Layer2Size', [200 600], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-3, 1e-2], 'Transform', 'log')
    optimizableVariable('Lambda', [1e-6, 1e-4], 'Transform', 'log')
    optimizableVariable('DropoutP', [0.10, 0.40])
    optimizableVariable('MiniBatchSize', {'1024', '2048', '4096', '8192', '16384'}, 'Type', 'categorical')
    ];

% --- MEMORY/COMPUTE CONSTRAINT ---
% Defines the logic to prevent "Huge Network + Tiny Batch" stalls
optimizedMemoryConstraint = @(x) applyTieredMemoryBudget(x);

%% 3. MAIN DISCOVERY LOOP
outerFoldResults = struct('BestHyperparams', {}, 'TestMetrics', {}, 'Network', {});

for outerFold = 1:K_outer
    fprintf('\n>>> Outer Fold %d/%d <<<\n', outerFold, K_outer);

    % Outer Masks
    maskTrainPool = (FT.FoldOuter ~= outerFold) & (FT.FoldOuter > 0) & maskNCV;
    maskTest      = (FT.FoldOuter == outerFold) & maskNCV;

    % --- EXPLICIT LEAKAGE CHECK ---
    fprintf('  [QA] Performing SourceID Leakage Check...\n');
    T_CheckTrain = FT(maskTrainPool, {'SourceID', 'AHM_7_Class'});
    T_CheckTest  = FT(maskTest,      {'SourceID', 'AHM_7_Class'});

    [isClean, leakReport] = LeakageCheck(T_CheckTrain, T_CheckTest);

    if ~isClean
        error('FATAL: SourceID Leakage detected between Outer Fold %d Train and Test sets! Halting.', outerFold);
    else
        fprintf('  [QA] Leakage Check PASSED. (Ratio Heli/Other: Train=%.2f, Test=%.2f)\n', ...
            leakReport.Ratio_H_O(1), leakReport.Ratio_H_O(2));
    end

    % --- GENERATE INNER FOLDS ---
    fprintf('  Generating Inner Folds for optimization...\n');
    innerFolds = CV_Audio(FT, maskTrainPool, K_inner);

    % --- 3.A OPTIMIZATION (Using BayesOpt Wrapper) ---
    objFcn = @(params) objFcn_Direct(params, FT, maskTrainPool, innerFolds, predictorNames, responseName, K_inner, fixedNetParams, refClassNames, TARGET_BALANCE);

    % Pass constraint function to BayesOpt
    results = BayesOpt(objFcn, optimVars, numBayesoptEvaluations, [outerFold, 0], optimizedMemoryConstraint);

    bestParams = results.minObjective; % Standard bayesopt result struct property
    if isempty(bestParams)
        if istable(results.XAtMinObjective)
            bestParams = results.XAtMinObjective;
        else
            % If BayesOpt returns struct with history, find min manually
            [~, idx] = min(results.ObjectiveTrace);
            bestParams = results.XTrace(idx, :);
        end
    else
        bestParams = results.XAtMinObjective;
    end

    outerFoldResults(outerFold).BestHyperparams = bestParams;

    % --- 3.B FINAL MODEL TRAINING ---
    fprintf('  Training Final Model for Fold %d...\n', outerFold);

    % 1. Prepare Data
    maskTrainFinal = collectHeliWithAugments(FT, maskTrainPool);

    X_train_raw = FT{maskTrainFinal, predictorNames};
    Y_train_raw = string(FT{maskTrainFinal, responseName});

    % 2. Balance
    [X_train_bal, Y_train_str] = ClassBalance(X_train_raw, Y_train_raw, TARGET_BALANCE);
    Y_train = categorical(Y_train_str, refClassNames);

    X_test = FT{maskTest, predictorNames};
    Y_test = categorical(string(FT{maskTest, responseName}), refClassNames);

    % 3. Pipeline
    [~, mu, sigma] = zscore(X_train_raw);
    sigma(sigma==0)=1;
    X_train_norm = (X_train_bal - mu) ./ sigma;
    X_test_norm  = (X_test - mu) ./ sigma;

    % 4. Architecture
    if iscategorical(bestParams.MiniBatchSize)
        currBatchSize = str2double(string(bestParams.MiniBatchSize));
    else
        currBatchSize = bestParams.MiniBatchSize;
    end

    inputSize = size(X_train_norm, 2);

    layers = createNeuralNetworkLayers(inputSize, bestParams, fixedNetParams, []);

    % 5. Train
    opts = trainingOptions(fixedNetParams.Solver, ...
        'MaxEpochs', fixedNetParams.MaxEpochs, ...
        'MiniBatchSize', currBatchSize, ...
        'InitialLearnRate', bestParams.InitialLearnRate, ...
        'L2Regularization', bestParams.Lambda, ...
        'Shuffle', 'every-epoch', 'Verbose', 0, ...
        'ExecutionEnvironment', fixedNetParams.ExecutionEnvironment, ...
        'ValidationData', {X_test_norm, Y_test}, ...
        'ValidationFrequency', fixedNetParams.ValidationFrequency, ...
        'ValidationPatience', fixedNetParams.ValidationPatience, ...
        'OutputNetwork', 'best-validation-loss');

    [net, info] = TrainNetwork(X_train_norm, Y_train, layers, opts);

    % 6. Metrics
    [Y_pred, Y_scores] = classify(net, X_test_norm);

    metrics = CalculateMetrics(Y_test, Y_pred, Y_scores, char(fixedNetParams.ClassNames(1)), fixedNetParams.ClassNames);

    outerFoldResults(outerFold).TestMetrics = metrics;
    outerFoldResults(outerFold).Network = net;

    fprintf('  Fold %d Test Results: Acc=%.4f, MacroF1=%.4f, HeliF1=%.4f\n', ...
        outerFold, metrics.Accuracy, mean(metrics.PerClassF1), metrics.PositiveF1_score);
end

%% 4. SAVE
save(fullfile(resultsDir, 'Discovery_Results.mat'), ...
    'outerFoldResults', 'fixedNetParams', 'predictorNames', 'responseName', '-v7.3');
fprintf('\nDone. Results in %s\n', resultsDir);
diary off;

%% LOCAL FUNCTIONS

function tf = applyTieredMemoryBudget(x)
% applyTieredMemoryBudget Prevents GPU stalls by enforcing Batch/Size ratios.
% Vectorized for bayesopt table compatibility.
%
% Inputs:
%   x: Table of hyperparameters (can be multiple rows)
% Outputs:
%   tf: Logical column vector

% 1. Handle Categorical MiniBatchSize (Robust Vectorized Conversion)
if iscategorical(x.MiniBatchSize) || isstring(x.MiniBatchSize)
    bs = str2double(string(x.MiniBatchSize));
elseif iscell(x.MiniBatchSize)
    bs = str2double(string(x.MiniBatchSize));
else
    bs = x.MiniBatchSize;
end

% 2. Extract numeric arrays
l1 = x.Layer1Size;
l2 = x.Layer2Size;
totalNodes = l1 + l2;

% 3. Initialize valid mask (Default True)
tf = true(height(x), 1);

% 4. Apply Constraints (Element-wise Logic)

% Constraint A: Small Batch (<= 2048) cannot have Massive Networks (> 1200)
% Invalid condition: (bs <= 2048) & (totalNodes > 1200)
maskA = (bs <= 2048) & (totalNodes > 1200);
tf(maskA) = false;

% Constraint B: Huge L1 (> 1000) requires Parallelism (Batch >= 4096)
% Invalid condition: (l1 > 1000) & (bs < 4096)
maskB = (l1 > 1000) & (bs < 4096);
tf(maskB) = false;

% Constraint C: Very Small Batch (1024) stricter limit (> 800 nodes)
% Invalid condition: (bs <= 1024) & (totalNodes > 800)
maskC = (bs <= 1024) & (totalNodes > 800);
tf(maskC) = false;

end