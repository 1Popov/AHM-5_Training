%% AHM_Deep_Diagnostics.m
% Purpose: Forensic analysis of the Final Model vs. Ensemble on the ENTIRE dataset.
% breakdown by 7 Classes and Augmentation status.

clear; clc; close all;

% --- CONFIGURATION ---
% Path to the Production Asset you just created
assetPath = '.\Results\Production_27-Nov-2025_09_58\AHM_Final_Model_Asset.mat';

% Path to the Data
ftPath = 'Main_4C_Final.mat';

addpath('C:\NextCloud\Work Projects\AHM\4_FeatureCleaning');
addpath('C:\NextCloud\Work Projects\AHM\Utilities');

%% 1. LOAD RESOURCES
fprintf('Loading Asset: %s...\n', assetPath);
Asset = load(assetPath);

% Map Legacy Variable Names to Script Variables
if isfield(Asset, 'production_model')
    finalNet = Asset.production_model;
    normParams = Asset.production_norm_params;
    threshold = Asset.production_threshold;
    refClassNames = cellstr(Asset.fixedNetParams.ClassNames);
elseif isfield(Asset, 'Network')
    finalNet = Asset.Network;
    normParams = Asset.Normalization;
    threshold = Asset.OptimalThreshold;
    refClassNames = cellstr(Asset.ClassNames);
else
    error('Unknown Asset Format. Could not find network model.');
end

cv_models = Asset.cv_models;
predictorNames = Asset.predictorNames;

fprintf('Loading Data: %s...\n', ftPath);
load(ftPath, 'FT');

%% 2. PREPARE FULL DATA MATRIX
fprintf('Preparing Full Dataset for Inference...\n');
X_raw = FT{:, predictorNames};

% Normalize using PRODUCTION statistics
X_norm = (X_raw - normParams.mu) ./ normParams.sigma;

% Identify Helicopter Class Index
posIdx = find(strcmp(refClassNames, "Helicopter"));

%% 3. INFERENCE - FINAL MODEL
fprintf('Running Final Model (Single) on %d rows...\n', height(FT));
[~, scores_final] = classify(finalNet, X_norm);
probs_final = scores_final(:, posIdx);

%% 4. INFERENCE - ENSEMBLE
fprintf('Running Ensemble (5 Models) on %d rows...\n', height(FT));
probs_accum = zeros(height(FT), 1);

for k = 1:5
    % fprintf('  Forward pass Model %d/5...\n', k);
    [~, sc] = classify(cv_models{k}, X_norm);
    probs_accum = probs_accum + sc(:, posIdx);
end
probs_ensemble = probs_accum / 5;

%% 5. DEEP DIVE ANALYSIS
fprintf('\n=== FORENSIC BREAKDOWN (Threshold = %.2f) ===\n', threshold);

% Define categories
categories_to_test = {
    'Heli (Original)',   (FT.AHM_7_Class == "Helicopter" & ~FT.isAugment);
    'Heli (Augment)',    (FT.AHM_7_Class == "Helicopter" & FT.isAugment);
    'Aircraft',          (FT.AHM_7_Class == "Aircraft_Confuser");
    'Siren',             (FT.AHM_7_Class == "Siren_Confuser");
    'Vehicle',           (FT.AHM_7_Class == "Vehicle_Confuser");
    'Military',          (FT.AHM_7_Class == "Military_Confuser");
    'Silence',           (FT.AHM_7_Class == "Ambient_Silence");
    'General Noise',     (FT.AHM_7_Class == "General_Noise");
    };

% Calculation Loop
ResultsStruct = struct();
for i = 1:length(categories_to_test)
    catName = categories_to_test{i, 1};
    mask = categories_to_test{i, 2};
    n_items = nnz(mask);

    if n_items > 0
        trigs_sgl = sum(probs_final(mask) >= threshold);
        rate_sgl  = (trigs_sgl / n_items) * 100;

        trigs_ens = sum(probs_ensemble(mask) >= threshold);
        rate_ens  = (trigs_ens / n_items) * 100;
    else
        trigs_sgl = 0; rate_sgl = 0;
        trigs_ens = 0; rate_ens = 0;
    end

    ResultsStruct(i).Category = catName;
    ResultsStruct(i).Total_Count = n_items;
    ResultsStruct(i).Triggers_Single = trigs_sgl;
    ResultsStruct(i).Rate_Single = rate_sgl;
    ResultsStruct(i).Triggers_Ensemble = trigs_ens;
    ResultsStruct(i).Rate_Ensemble = rate_ens;
end

% --- PRINT FORMATTED TABLE ---
fprintf('\n%-20s | %12s | %12s | %12s | %12s | %12s |\n', ...
    'Category', 'Total Samples', 'Trig(Sgl)', 'Rate(Sgl) %', 'Trig(Ens)', 'Rate(Ens) %');
fprintf('---------------------|--------------|--------------|--------------|--------------|--------------|\n');
for i = 1:length(ResultsStruct)
    fprintf('%-20s | %12d | %12d | %11.2f%% | %12d | %11.2f%% |\n', ...
        ResultsStruct(i).Category, ...
        ResultsStruct(i).Total_Count, ...
        ResultsStruct(i).Triggers_Single, ...
        ResultsStruct(i).Rate_Single, ...
        ResultsStruct(i).Triggers_Ensemble, ...
        ResultsStruct(i).Rate_Ensemble);
end
fprintf('----------------------------------------------------------------------------------------------\n');

%% 6. CONFUSION MATRICES (Binary)
fprintf('\n>>> Generating Confusion Matrices <<<\n');

% Create Binary Labels (True)
true_labels_bin = repmat("Background", height(FT), 1);
true_labels_bin(FT.AHM_7_Class == "Helicopter") = "Helicopter";
true_labels_bin = categorical(true_labels_bin, ["Helicopter", "Background"]);

% Create Predictions (Single)
pred_sgl_bin = repmat("Background", height(FT), 1);
pred_sgl_bin(probs_final >= threshold) = "Helicopter";
pred_sgl_bin = categorical(pred_sgl_bin, ["Helicopter", "Background"]);

% Create Predictions (Ensemble)
pred_ens_bin = repmat("Background", height(FT), 1);
pred_ens_bin(probs_ensemble >= threshold) = "Helicopter";
pred_ens_bin = categorical(pred_ens_bin, ["Helicopter", "Background"]);

% Plot Single
fig1 = figure('Name', 'Single Model CM', 'Position', [100 100 600 500]);
cm1 = confusionchart(true_labels_bin, pred_sgl_bin);
cm1.Title = sprintf('Single Model (Th=%.2f) - Full Dataset', threshold);
cm1.RowSummary = 'row-normalized';
cm1.ColumnSummary = 'column-normalized';

% Plot Ensemble
fig2 = figure('Name', 'Ensemble Model CM', 'Position', [750 100 600 500]);
cm2 = confusionchart(true_labels_bin, pred_ens_bin);
cm2.Title = sprintf('Ensemble (5-Fold) (Th=%.2f) - Full Dataset', threshold);
cm2.RowSummary = 'row-normalized';
cm2.ColumnSummary = 'column-normalized';

%% 7. DETECTION RATE BAR CHART
figure('Name', 'Detection Rate', 'Position', [100 100 1200 600]);
cats = {ResultsStruct.Category};
vals_sgl = [ResultsStruct.Rate_Single];
vals_ens = [ResultsStruct.Rate_Ensemble];

b = bar(categorical(cats), [vals_sgl; vals_ens]');
ylabel('Alarm Trigger Rate (%)');
title(sprintf('Detection Rate per Class (Threshold %.2f)', threshold));
legend('Single Final Model', 'Ensemble (5-Fold)');
grid on;
ylim([0 105]);

% Add numeric labels
xtips1 = b(1).XEndPoints; ytips1 = b(1).YEndPoints;
labels1 = string(round(b(1).YData, 1)) + "%";
text(xtips1, ytips1, labels1,'HorizontalAlignment','center','VerticalAlignment','bottom', 'FontSize', 8);

xtips2 = b(2).XEndPoints; ytips2 = b(2).YEndPoints;
labels2 = string(round(b(2).YData, 1)) + "%";
text(xtips2, ytips2, labels2,'HorizontalAlignment','center','VerticalAlignment','bottom', 'FontSize', 8);

fprintf('\nDone. Check figures for matrices and charts.\n');