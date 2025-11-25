%% AHM_Discovery.m
% Stage 5 Discovery: Hyperparameter Tuning (7-Class) via Nested CV.
% STRICT CONFIGURATION: 5x5 Folds + SourceID Leakage Checks + Direct WST.
% UPDATED:
%   1. Includes Hold-Out Set Evaluation for every fold.
%   2. Expanded Hyperparameter Ranges based on Pilot Run (Capacity Increase).
%   3. Updated Compute Constraints for larger networks.

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

% --- 1.1 DEFINE DOMAINS ---
% NCV Domain: All non-augmented, valid segments designated for development
% Explicitly EXCLUDE "test" and "eval" splits
maskNCV = FT.legitForNCV & ~FT.isAugment & ~ismember(FT.OfficialSplit, ["test", "eval"]);

% Hold-Out Domain: Strictly unseen "test" (MAD) and "eval" (FSD50K) data
maskHeldOut = FT.legitForNCV & ~FT.isAugment & ismember(FT.OfficialSplit, ["test", "eval"]);

fprintf('  NCV Development Rows:     %d\n', nnz(maskNCV));
fprintf('  Held-Out Rows (Test/Eval): %d\n', nnz(maskHeldOut));

% --- 1.2 ASSIGN OUTER FOLDS (In-Memory) ---
fprintf('Assigning Outer Folds...\n');
FT.FoldOuter = zeros(height(FT), 1);

% A. Respect ESC50 Official Folds
maskESC = maskNCV & (FT.Dataset == "ESC50");
if any(maskESC)
    esc_splits = string(FT.OfficialSplit(maskESC));
    esc_folds = double(erase(esc_splits, "fold"));
    esc_folds(isnan(esc_folds)) = 0;
    FT.FoldOuter(maskESC) = esc_folds;
    fprintf('  Mapped %d ESC50 rows to official folds.\n', nnz(esc_folds > 0));
end

% B. Assign Folds for Rest (CV_Audio - Leak Proof)
maskRest = maskNCV & (FT.FoldOuter == 0);
if any(maskRest)
    fprintf('  Running CV_Audio on %d remaining rows...\n', nnz(maskRest));
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
numBayesoptEvaluations = 80;
TARGET_BALANCE = 30000;

% Fixed Network Params
fixedNetParams.Activations = 'relu';
fixedNetParams.Solver = 'adam';
fixedNetParams.MaxEpochs = 120;
fixedNetParams.Verbose = 0;
fixedNetParams.ExecutionEnvironment = 'gpu';
fixedNetParams.ValidationFrequency = 50;
fixedNetParams.ValidationPatience = 12;

% MASTER CLASS LIST - Enforce Strict Categorical Order
rawCats = categories(FT.(responseName));
fixedNetParams.ClassNames = categorical(rawCats, rawCats, 'Ordinal', false);
refClassNames = cellstr(fixedNetParams.ClassNames);

% --- OPTIMIZED SEARCH SPACE (RE-TUNED RANGES) ---
% Shifted based on Pilot Run hitting ceilings.
% Networks need more capacity for raw WST, and higher Dropout/Lambda to control it.
optimVars = [
    optimizableVariable('Layer1Size', [800 2000], 'Type', 'integer')         % Was [400 1200]
    optimizableVariable('Layer2Size', [400 1000], 'Type', 'integer')         % Was [200 600]
    optimizableVariable('InitialLearnRate', [5e-4, 5e-3], 'Transform', 'log')
    optimizableVariable('Lambda', [1e-5, 1e-2], 'Transform', 'log')          % Was [1e-6 1e-4]
    optimizableVariable('DropoutP', [0.30, 0.60])                            % Was [0.10 0.40]
    optimizableVariable('MiniBatchSize', {'1024', '2048', '4096', '8192', '16384'}, 'Type', 'categorical')
    ];

% --- MEMORY CONSTRAINT ---
% Updated logic to handle the new larger layer limits
optimizedMemoryConstraint = @(x) applyTieredMemoryBudget(x);

%% 3. MAIN DISCOVERY LOOP
outerFoldResults = struct('BestHyperparams', {}, 'TestMetrics', {}, 'HoldOutMetrics', {}, 'Network', {});

for outerFold = 1:K_outer
    fprintf('\n>>> Outer Fold %d/%d <<<\n', outerFold, K_outer);

    % Outer Masks
    maskTrainPool = (FT.FoldOuter ~= outerFold) & (FT.FoldOuter > 0) & maskNCV;
    maskTest      = (FT.FoldOuter == outerFold) & maskNCV;

    % --- EXPLICIT LEAKAGE CHECK ---
    fprintf('  [QA] Performing SourceID Leakage Check...\n');
    % Pass required columns for ESC50 logic
    T_CheckTrain = FT(maskTrainPool, {'SourceID', 'AHM_7_Class', 'Dataset', 'OriginalStem'});
    T_CheckTest  = FT(maskTest,      {'SourceID', 'AHM_7_Class', 'Dataset', 'OriginalStem'});

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

    % --- 3.A OPTIMIZATION ---
    objFcn = @(params) objFcn_Direct(params, FT, maskTrainPool, innerFolds, predictorNames, responseName, K_inner, fixedNetParams, refClassNames, TARGET_BALANCE);

    results = BayesOpt(objFcn, optimVars, numBayesoptEvaluations, [outerFold, 0], optimizedMemoryConstraint);

    if isempty(results)
        warning('BayesOpt returned empty results. Using fallback.');
        bestParams = table(1000, 500, 0.002, 1e-4, 0.40, categorical({'4096'}), ...
            'VariableNames', {'Layer1Size','Layer2Size','InitialLearnRate','Lambda','DropoutP','MiniBatchSize'});
    else
        bestParams = results.XAtMinObjective;
    end

    outerFoldResults(outerFold).BestHyperparams = bestParams;
    fprintf('  Best Hyperparameters for Fold %d found.\n', outerFold);
    disp(bestParams);

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

    % 3. Pipeline (Normalize)
    % CRITICAL: Stats derived ONLY from Training data
    [~, mu, sigma] = zscore(X_train_raw);
    sigma(sigma==0)=1;

    X_train_norm = (X_train_bal - mu) ./ sigma;
    X_test_norm  = (X_test - mu) ./ sigma;

    % 4. Train
    if iscategorical(bestParams.MiniBatchSize)
        currBatchSize = str2double(string(bestParams.MiniBatchSize));
    else
        currBatchSize = bestParams.MiniBatchSize;
    end

    layers = createNeuralNetworkLayers(size(X_train_norm,2), bestParams, fixedNetParams, []);
    opts = createTrainingOptions(bestParams, fixedNetParams, X_test_norm, Y_test);

    [net, ~] = TrainNetwork(X_train_norm, Y_train, layers, opts);

    % --- 3.C EVALUATION (NCV Test) ---
    [Y_pred, Y_scores] = classify(net, X_test_norm);
    metrics = CalculateMetrics(Y_test, Y_pred, Y_scores, char(fixedNetParams.ClassNames(1)), fixedNetParams.ClassNames);
    outerFoldResults(outerFold).TestMetrics = metrics;

    % --- 3.D EVALUATION (Hold-Out) ---
    % This evaluates the model trained on Fold X against the global unseen Test set
    fprintf('  Evaluating on Hold-Out Set (Unseen Data)...\n');

    X_ho = FT{maskHeldOut, predictorNames};
    Y_ho = categorical(string(FT{maskHeldOut, responseName}), refClassNames);

    % Normalize Hold-Out using THIS fold's training stats (mu, sigma)
    % This ensures valid comparison
    X_ho_norm = (X_ho - mu) ./ sigma;

    [Y_ho_pred, Y_ho_scores] = classify(net, X_ho_norm);
    ho_metrics = CalculateMetrics(Y_ho, Y_ho_pred, Y_ho_scores, char(fixedNetParams.ClassNames(1)), fixedNetParams.ClassNames);
    outerFoldResults(outerFold).HoldOutMetrics = ho_metrics;
    outerFoldResults(outerFold).Network = net;

    fprintf('  Fold %d NCV Test: Acc=%.4f, HeliF1=%.4f, HeliAUC=%.4f\n', ...
        outerFold, metrics.Accuracy, metrics.PositiveF1_score, metrics.PositiveAUC_ROC);
    fprintf('  Fold %d Hold-Out: Acc=%.4f, HeliF1=%.4f, HeliAUC=%.4f\n', ...
        outerFold, ho_metrics.Accuracy, ho_metrics.PositiveF1_score, ho_metrics.PositiveAUC_ROC);

    % Memory Cleanup
    reset(gpuDevice);
end

%% 4. SAVE
save(fullfile(resultsDir, 'Discovery_Results.mat'), ...
    'outerFoldResults', 'fixedNetParams', 'predictorNames', 'responseName', '-v7.3');
fprintf('\nDone. Results in %s\n', resultsDir);
diary off;

%% LOCAL FUNCTIONS

function tf = applyTieredMemoryBudget(x)
% applyTieredMemoryBudget Vectorized constraint function
% Prevents stall by ensuring Batch Size scales with Network Size
if iscategorical(x.MiniBatchSize) || isstring(x.MiniBatchSize)
    bs = str2double(string(x.MiniBatchSize));
elseif iscell(x.MiniBatchSize)
    bs = str2double(string(x.MiniBatchSize));
else
    bs = x.MiniBatchSize;
end

l1 = x.Layer1Size;
l2 = x.Layer2Size;
totalNodes = l1 + l2;

tf = true(height(x), 1);

% Update constraints for new larger ranges [800, 2000]

% A: Small Batch (<=2048) forbids Massive Networks (> 2000 total nodes)
% Increased threshold from 1200 to 2000 because newer GPUs can handle it if batch isn't tiny
maskA = (bs <= 2048) & (totalNodes > 2000);
tf(maskA) = false;

% B: Huge L1 (> 1500) requires Parallelism (Batch >= 4096)
maskB = (l1 > 1500) & (bs < 4096);
tf(maskB) = false;

% C: Very Small Batch (1024) strict limit (> 1200 nodes)
maskC = (bs <= 1024) & (totalNodes > 1200);
tf(maskC) = false;
end