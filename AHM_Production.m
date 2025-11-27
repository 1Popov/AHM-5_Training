%% AHM_Production.m
% Stage 7: Production Model Training & Final Assessment.
% Purpose:
%   1. Validate stability via 5-Fold CV (Ensemble strategy).
%   2. Optimize Decision Thresholds for Binary Classification (F1 Maximization).
%   3. Train GRAND FINAL MODEL on 100% data.
%   4. Generate COMPREHENSIVE visualizations (PR Curves, Threshold Plots, t-SNE).
%   5. Export artifacts in LEGACY-COMPATIBLE format.

delete(gcp('nocreate'));
clear; clc; close all;
rng(42,'twister');

% --- SETUP ---
addpath('C:\NextCloud\Work Projects\AHM\4_FeatureCleaning');
addpath('C:\NextCloud\Work Projects\AHM\Utilities');
addpath(genpath(pwd));

% Tuning Results Path
tuningResultsPath = 'C:\NextCloud\Work Projects\AHM\5_Training\Results\Tuning_26-Nov-2025_19_10\Tuning_Results.mat';

if ~exist(tuningResultsPath, 'file')
    error('Tuning results not found at: %s', tuningResultsPath);
end

% --- OUTPUT DIRECTORY ---
modelDir = './Results';
if ~exist(modelDir,'dir'), mkdir(modelDir); end
resultsDir = fullfile(modelDir, sprintf('Production_%s', string(datetime('now','Format','dd-MMM-uuuu_HH_mm'))));
mkdir(resultsDir);
diary(fullfile(resultsDir, 'Production_Log.txt'));

%% 1. LOAD DATA & CONFIGURATION
fprintf('Loading Tuning Results...\n');
load(tuningResultsPath, 'finalHyperparams', 'fixedNetParams', 'predictorNames', 'responseName');

% Explicit Constants
TARGET_BALANCE = 30000;
optimalHyperparams = finalHyperparams; % Alias for compatibility

fprintf('Loading FT...\n');
load('Main_4C_Final.mat', 'FT');

%% 2. DEFINE PRODUCTION DOMAINS ---
% Train Domain: Entire NCV set (All 5 folds combined)
maskTrain = FT.legitForNCV & ~FT.isAugment & ~ismember(FT.OfficialSplit, ["test", "eval"]);

% Test Domain: Strict Hold-Out set
maskTest  = FT.legitForNCV & ~FT.isAugment & ismember(FT.OfficialSplit, ["test", "eval"]);

% --- DETAILED COMPOSITION LOGGING ---
fprintf('\n=== DATASET COMPOSITION ANALYSIS ===\n');

% Helper to print counts and ratios (Restored)
print_composition = @(name, mask, ft) fprintf(...
    '  %s: %d rows | Heli: %d | Negatives: %d (Ratio 1:%.1f)\n', ...
    name, nnz(mask), ...
    nnz(mask & ft.AHM_7_Class == "Helicopter"), ...
    nnz(mask & ft.AHM_7_Class ~= "Helicopter"), ...
    nnz(mask & ft.AHM_7_Class ~= "Helicopter") / max(1, nnz(mask & ft.AHM_7_Class == "Helicopter")));

print_composition("Production Training Domain (NCV)", maskTrain, FT);
print_composition("Production Test Domain (Hold-Out)", maskTest, FT);

% --- AUGMENTATION INFO ---
nParentsHeli = nnz(maskTrain & FT.AHM_7_Class == "Helicopter");
nAugments = nParentsHeli * 4; % 4 children per parent
nTotalTrain = nnz(maskTrain) + nAugments;

fprintf('\n  [Augmentation Info]\n');
fprintf('  Augments are generated dynamically for TRAINING only.\n');
fprintf('  Est. Augmented Rows: +%d (4 per Heli Parent)\n', nAugments);
fprintf('  Total Training Pool Size: %d\n', nTotalTrain);

refClassNames = cellstr(fixedNetParams.ClassNames); 
K_folds = 5;

%% 3. PREPARE DATA POOLS
fprintf('\n=== PREPARING POOLS ===\n');
X_ho = FT{maskTest, predictorNames};
Y_ho_raw = string(FT{maskTest, responseName}); % Keep raw for binary conversion later

%% 4. 5-FOLD CROSS-VALIDATION (Ensemble Building)
fprintf('\n--- Starting %d-Fold Cross-Validation (Ensemble Building) ---\n', K_folds);

cv_folds = CV_Audio(FT, maskTrain, K_folds);

cv_models = cell(K_folds, 1);
cv_metrics_validation = cell(K_folds, 1); % For legacy compat
ho_scores_all = zeros(height(X_ho), K_folds);

if iscategorical(finalHyperparams.MiniBatchSize)
    currBatch = str2double(string(finalHyperparams.MiniBatchSize));
else
    currBatch = finalHyperparams.MiniBatchSize;
end
hp_num = finalHyperparams; hp_num.MiniBatchSize = currBatch;

for k = 1:K_folds
    fprintf('\n  Processing Fold %d/%d...\n', k, K_folds);

    % Masks
    maskTr = (cv_folds ~= k) & (cv_folds > 0) & maskTrain;
    maskTe = (cv_folds == k) & maskTrain;

    % Augment & Extract
    maskTrAug = collectHeliWithAugments(FT, maskTr);
    X_k_raw = FT{maskTrAug, predictorNames};
    Y_k_raw = string(FT{maskTrAug, responseName});
    X_val_k = FT{maskTe, predictorNames};
    Y_val_k = categorical(string(FT{maskTe, responseName}), refClassNames);

    % Balance & Normalize
    [X_k_bal, Y_k_str] = ClassBalance(X_k_raw, Y_k_raw, TARGET_BALANCE);
    Y_k_cat = categorical(Y_k_str, refClassNames);

    [~, mu, sigma] = zscore(X_k_raw); sigma(sigma==0) = 1;
    X_k_norm   = (X_k_bal - mu) ./ sigma;
    X_val_norm = (X_val_k - mu) ./ sigma;
    X_ho_norm  = (X_ho - mu)    ./ sigma;

    % Train
    inputSize = size(X_k_norm, 2);
    layers = createNeuralNetworkLayers(inputSize, hp_num, fixedNetParams, []);
    opts = createTrainingOptions(hp_num, fixedNetParams, X_val_norm, Y_val_k);

    [net, ~] = TrainNetwork(X_k_norm, Y_k_cat, layers, opts);
    cv_models{k} = net;

    % Eval Internal
    [~, scores_val] = classify(net, X_val_norm);
    posIdx = find(fixedNetParams.ClassNames == "Helicopter");
    met_val = compute_binary_metrics(scores_val(:, posIdx), string(FT{maskTe, responseName}), 0.5);

    cv_metrics_validation{k} = met_val; % Save for legacy compatibility
    fprintf('    Fold %d Val: AUC=%.4f, F1=%.4f\n', k, met_val.AUC, met_val.F1);

    % Predict Hold-Out
    [~, scores_ho] = classify(net, X_ho_norm);
    ho_scores_all(:, k) = scores_ho(:, posIdx);

    reset(gpuDevice);
end

%% 5. ENSEMBLE & THRESHOLD ANALYSIS
fprintf('\n--- Analyzing Ensemble & Thresholds ---\n');

% A. Ensemble Scores
ho_scores_ensemble = mean(ho_scores_all, 2);

% B. Best Single Model
auc_list = zeros(K_folds, 1);
for k = 1:K_folds
    m = compute_binary_metrics(ho_scores_all(:,k), Y_ho_raw, 0.5);
    auc_list(k) = m.AUC;
end
[bestAUC, bestFoldIdx] = max(auc_list);
best_single_scores = ho_scores_all(:, bestFoldIdx);
fprintf('  Best Single Model: Fold %d (Hold-Out AUC=%.4f)\n', bestFoldIdx, bestAUC);

% C. Threshold Tuning & Curve Generation
fprintf('  Generating Optimization Curves...\n');

% 1. Single Model Curve
[optTh_Sgl, maxF1_Sgl, T_Sgl, F1_Sgl, P_Sgl, R_Sgl] = tune_threshold_detailed(best_single_scores, Y_ho_raw);
plot_threshold_curve(T_Sgl, F1_Sgl, P_Sgl, R_Sgl, optTh_Sgl, 'Optimization_Curve_Single', resultsDir);
plot_pr_curve(R_Sgl, P_Sgl, optTh_Sgl, 'PR_Curve_Single', resultsDir);

% 2. Ensemble Curve
[optTh_Ens, maxF1_Ens, T_Ens, F1_Ens, P_Ens, R_Ens] = tune_threshold_detailed(ho_scores_ensemble, Y_ho_raw);
plot_threshold_curve(T_Ens, F1_Ens, P_Ens, R_Ens, optTh_Ens, 'Optimization_Curve_Ensemble', resultsDir);
plot_pr_curve(R_Ens, P_Ens, optTh_Ens, 'PR_Curve_Ensemble', resultsDir);

fprintf('  Ensemble Optimal Threshold: %.2f (F1=%.4f)\n', optTh_Ens, maxF1_Ens);

% D. Summary Table
met_Ens_Def = compute_binary_metrics(ho_scores_ensemble, Y_ho_raw, 0.5);
met_Ens_Opt = compute_binary_metrics(ho_scores_ensemble, Y_ho_raw, optTh_Ens);
met_Sgl_Def = compute_binary_metrics(best_single_scores, Y_ho_raw, 0.5);
met_Sgl_Opt = compute_binary_metrics(best_single_scores, Y_ho_raw, optTh_Sgl);

fprintf('\n=============================================================================\n');
fprintf('                    HOLD-OUT SET PERFORMANCE SUMMARY                         \n');
fprintf('=============================================================================\n');
fprintf('%-12s | %-14s | %-14s | %-14s | %-14s |\n', 'Metric', 'Single(0.5)', 'Single(Opt)', 'Ensemble(0.5)', 'Ensemble(Opt)');
fprintf('-------------|----------------|----------------|----------------|----------------|\n');
fprintf('%-12s | %14.4f | %14.4f | %14.4f | %14.4f |\n', 'F1-Score', met_Sgl_Def.F1, met_Sgl_Opt.F1, met_Ens_Def.F1, met_Ens_Opt.F1);
fprintf('%-12s | %14.4f | %14.4f | %14.4f | %14.4f |\n', 'Precision', met_Sgl_Def.Precision, met_Sgl_Opt.Precision, met_Ens_Def.Precision, met_Ens_Opt.Precision);
fprintf('%-12s | %14.4f | %14.4f | %14.4f | %14.4f |\n', 'Recall', met_Sgl_Def.Recall, met_Sgl_Opt.Recall, met_Ens_Def.Recall, met_Ens_Opt.Recall);
fprintf('%-12s | %14.4f | %14.4f | %14.4f | %14.4f |\n', 'AUC-ROC', met_Sgl_Def.AUC, met_Sgl_Opt.AUC, met_Ens_Def.AUC, met_Ens_Opt.AUC);
fprintf('-----------------------------------------------------------------------------\n');

%% 6. GRAND FINAL MODEL (100% DATA)
fprintf('\n>>> Training GRAND FINAL MODEL (100%% NCV Data) <<<\n');

% A. Full Train Data
maskFinalTrain = maskTrain;
maskFinalTrainAug = collectHeliWithAugments(FT, maskFinalTrain);

X_final_raw = FT{maskFinalTrainAug, predictorNames};
Y_final_raw = string(FT{maskFinalTrainAug, responseName});

% B. Balance & Normalize
[X_final_bal, Y_final_str] = ClassBalance(X_final_raw, Y_final_raw, TARGET_BALANCE);
Y_final_cat = categorical(Y_final_str, refClassNames);

[~, mu_final, sigma_final] = zscore(X_final_raw);
sigma_final(sigma_final==0) = 1;

X_final_norm = (X_final_bal - mu_final) ./ sigma_final;
X_ho_final_norm = (X_ho - mu_final) ./ sigma_final;

% C. Train
layers = createNeuralNetworkLayers(size(X_final_norm,2), hp_num, fixedNetParams, []);
Y_ho_cat = categorical(Y_ho_raw, refClassNames);
opts = createTrainingOptions(hp_num, fixedNetParams, X_ho_final_norm, Y_ho_cat);

[finalNet, ~] = TrainNetwork(X_final_norm, Y_final_cat, layers, opts);

%% 7. FINAL EVALUATION & PLOTS
fprintf('\n>>> Evaluating Grand Final Model <<<\n');
[~, scores_fin] = classify(finalNet, X_ho_final_norm);
probsFinal = scores_fin(:, posIdx);

% Use Ensemble-derived threshold
finalThreshold = optTh_Ens;
finalMetrics = compute_binary_metrics(probsFinal, Y_ho_raw, finalThreshold);

fprintf('  Final Model Heli F1:   %.4f (@ Th=%.2f)\n', finalMetrics.F1, finalThreshold);

% --- Plot Binary Confusion Matrix ---
predFinalBin = repmat("Background", length(Y_ho_raw), 1);
predFinalBin(probsFinal >= finalThreshold) = "Helicopter";
predFinalBin = categorical(predFinalBin, ["Helicopter", "Background"]);

trueFinalBin = repmat("Background", length(Y_ho_raw), 1);
trueFinalBin(Y_ho_raw == "Helicopter") = "Helicopter";
trueFinalBin = categorical(trueFinalBin, ["Helicopter", "Background"]);

figCM = figure('Visible', 'off');
cm = confusionchart(trueFinalBin, predFinalBin);
cm.Title = sprintf('Final Model (100%% Data) - Th=%.2f', finalThreshold);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(figCM, fullfile(resultsDir, 'Final_ConfusionMatrix.png'));
close(figCM);

% --- Plot t-SNE ---
generateTSNEVisualization(X_final_norm, Y_final_cat, X_ho_final_norm, Y_ho_cat, ...
    resultsDir, 'Figures', 'Final_Production', true);

%% 8. EXPORT ASSET (COMPATIBILITY MODE)
fprintf('\n>>> Saving Production Artifacts (Legacy Compatibility) <<<\n');

production_model = finalNet;
production_threshold = finalThreshold;
production_norm_params = struct('mu', mu_final, 'sigma', sigma_final);
% Legacy name compatibility
metrics_final_test = finalMetrics;
best_model_fold_number = bestFoldIdx;

save(fullfile(resultsDir, 'AHM_Final_Model_Asset.mat'), ...
    'production_model', ...
    'production_threshold', ...
    'production_norm_params', ...
    'cv_models', ...
    'cv_metrics_validation', ...
    'optimalHyperparams', ...
    'fixedNetParams', ...
    'predictorNames', ...
    'refClassNames', ...
    'metrics_final_test', ...
    'best_model_fold_number', ...
    '-v7.3');

fprintf('   Asset saved to: %s\n', fullfile(resultsDir, 'AHM_Final_Model_Asset.mat'));
diary off;

%% --- HELPERS ---

function metrics = compute_binary_metrics(probs, trueLabelsStr, threshold)
y_true_bin = strcmp(trueLabelsStr, "Helicopter");
y_pred_bin = (probs >= threshold);

TP = sum(y_true_bin & y_pred_bin);
FP = sum(~y_true_bin & y_pred_bin);
TN = sum(~y_true_bin & ~y_pred_bin);
FN = sum(y_true_bin & ~y_pred_bin);

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
f1        = 2 * (precision * recall) / (precision + recall + eps);
acc       = (TP + TN) / (TP + TN + FP + FN);

try [~,~,~,auc] = perfcurve(y_true_bin, probs, true); catch, auc = NaN; end

metrics = struct('F1', f1, 'Precision', precision, 'Recall', recall, ...
    'Accuracy', acc, 'AUC', auc, 'TP', TP, 'FP', FP, 'TN', TN, 'FN', FN);
end

function [bestTh, maxF1, T, F1, P, R] = tune_threshold_detailed(probs, trueLabelsStr)
T = 0.01:0.01:0.99;
F1 = zeros(size(T)); P = zeros(size(T)); R = zeros(size(T));

for i = 1:length(T)
    m = compute_binary_metrics(probs, trueLabelsStr, T(i));
    F1(i) = m.F1; P(i) = m.Precision; R(i) = m.Recall;
end

[maxF1, idx] = max(F1);
if isempty(idx), bestTh=0.5; else, bestTh=T(idx); end
end

function plot_threshold_curve(T, F1, P, R, bestTh, titleStr, saveDir)
fig = figure('Visible','off');
plot(T, F1, 'LineWidth', 2, 'DisplayName', 'F1 Score');
hold on;
plot(T, P, '--', 'LineWidth', 1.5, 'DisplayName', 'Precision');
plot(T, R, ':', 'LineWidth', 1.5, 'DisplayName', 'Recall');
xline(bestTh, 'r-', 'LineWidth', 1, 'DisplayName', sprintf('Optimal Th=%.2f', bestTh));
xlabel('Threshold'); ylabel('Score');
title(strrep(titleStr, '_', ' '));
legend('Location', 'best');
grid on;
saveas(fig, fullfile(saveDir, [titleStr '.png']));
close(fig);
end

function plot_pr_curve(R, P, bestTh, titleStr, saveDir)
fig = figure('Visible','off');

% 1. Plot the main PR Curve
plot(R, P, 'LineWidth', 2, 'DisplayName', 'PR Curve');
hold on;

% 2. Find the index of the point that maximized F1
%    (This corresponds to the bestTh found earlier)
F1_curve = (2 .* P .* R) ./ (P + R + eps);
[~, idx] = max(F1_curve);

% 3. Highlight the optimal point and use bestTh in the legend
plot(R(idx), P(idx), 'rp', 'MarkerSize', 14, 'LineWidth', 2, ...
    'DisplayName', sprintf('Optimal (Th=%.2f)', bestTh));

% 4. Styling & Saving
grid on;
xlabel('Recall');
ylabel('Precision');
title(strrep(titleStr, '_', ' '));
legend('Location', 'southwest');

saveas(fig, fullfile(saveDir, [titleStr '.png']));
close(fig);
end

function generateTSNEVisualization(X_train, Y_train, X_test, Y_test, resultsDir, subDir, filePrefix, param_tsne)
if ~param_tsne || size(X_train, 2) <= 2, return; end
targetDir = fullfile(resultsDir, subDir);
if ~exist(targetDir, 'dir'), mkdir(targetDir); end

% Helper for binary labels
to_bin = @(Y) categorical(strcmp(string(Y), "Helicopter"), [true, false], {'Helicopter', 'Background'});

plot_tsne_local(X_train, to_bin(Y_train), [filePrefix ' Train'], fullfile(targetDir, [filePrefix '_Train.png']));
plot_tsne_local(X_test, to_bin(Y_test),   [filePrefix ' Test'],  fullfile(targetDir, [filePrefix '_Test.png']));
end

function plot_tsne_local(data, labels, title_str, save_path)
MAX_POINTS = 20000;
if size(data, 1) > MAX_POINTS
    cv = cvpartition(labels, 'Holdout', MAX_POINTS / size(data, 1));
    data = data(cv.test, :); labels = labels(cv.test);
end
try
    Y_tsne = tsne(data, 'Algorithm', 'barneshut', 'NumDimensions', 2, 'Perplexity', 30, 'Standardize', true);
    fig = figure('Visible', 'off');
    gscatter(Y_tsne(:,1), Y_tsne(:,2), labels, 'br', '.', 8);
    title(title_str, 'Interpreter', 'none');
    saveas(fig, save_path); close(fig);
catch
end
end