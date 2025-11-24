function [objective, constraints, UserData] = objFcn_Direct(hyperparams, FT, maskTrainPool, innerFolds, predictorNames, responseName, K_inner, fixedNetParams, refClassNames, targetBalance)
% objFcn_Direct: Inner CV loop for Bayesian Optimization.
%
% Purpose:
%   Evaluates a specific set of hyperparameters by training K_inner models.
%   Returns the mean (1 - F1_Score) as the minimization objective.
%
% Inputs:
%   hyperparams:      Table row from bayesopt containing current parameters
%   FT:               Full Feature Table
%   maskTrainPool:    Logical mask of the current Outer Training Pool
%   innerFolds:       Integer vector matching maskTrainPool (0=skip, 1..K=fold)
%   predictorNames:   Cell array of feature names (90 WST)
%   responseName:     Name of the label column
%   K_inner:          Number of inner folds
%   fixedNetParams:   Struct of static network options (Epochs, Patience, etc.)
%   refClassNames:    Cellstr defining the strict categorical order of classes
%   targetBalance:    Integer target for physical class balancing
%
% Outputs:
%   objective:        Scalar float (Minimization target)
%   constraints:      [] (Handled by BayesOpt wrapper)
%   UserData:         Struct containing per-fold metrics for stability analysis

inner_objectives = ones(K_inner, 1); % Initialize with worst case (1.0)
inner_metrics_cell = cell(K_inner, 1);

% --- Parameter Pre-processing ---
% Convert categorical MiniBatchSize to numeric for trainingOptions
if iscategorical(hyperparams.MiniBatchSize)
    currBatchSize = str2double(string(hyperparams.MiniBatchSize));
else
    currBatchSize = hyperparams.MiniBatchSize;
end

% Create a numeric struct copy for layer creation helper
hyperparams_num = hyperparams;
hyperparams_num.MiniBatchSize = currBatchSize;

% --- Inner Cross-Validation Loop ---
for k = 1:K_inner
    % A. Define Splits based on Pre-Calculated Folds
    % Parents in Train Pool NOT in current fold k are Training
    maskInnerTrainParents = maskTrainPool & (innerFolds ~= k) & (innerFolds > 0);
    % Parents in Train Pool IN current fold k are Validation
    maskInnerVal          = maskTrainPool & (innerFolds == k);

    try
        % B. Augmentation (Apply ONLY to Training Set)
        % Dynamic expansion: Parents -> Parents + 4x Children
        maskInnerTrainFinal = collectHeliWithAugments(FT, maskInnerTrainParents);

        % C. Data Extraction
        X_train_raw = FT{maskInnerTrainFinal, predictorNames};
        Y_train_raw = string(FT{maskInnerTrainFinal, responseName});

        % Validation set is NEVER augmented/balanced
        X_val_raw   = FT{maskInnerVal, predictorNames};
        Y_val_raw   = string(FT{maskInnerVal, responseName});

        % D. Physical Class Balancing
        [X_train_bal, Y_train_str] = ClassBalance(X_train_raw, Y_train_raw, targetBalance);

        % --- STRICT CATEGORICAL ENFORCEMENT ---
        % Ensure labels match the network's Output Layer expectation exactly
        Y_train_cat = categorical(Y_train_str, refClassNames);
        Y_val_cat   = categorical(Y_val_raw, refClassNames);

        % E. Normalization (Z-Score)
        % Fit statistics on BALANCED TRAIN, apply to VAL
        [~, mu, sigma] = zscore(X_train_raw);
        sigma(sigma == 0) = 1; % Guard against constant features

        X_train_norm = (X_train_bal - mu) ./ sigma;
        X_val_norm   = (X_val_raw - mu) ./ sigma;

        % F. Network Architecture Construction
        inputSize = size(X_train_norm, 2);
        layers = createNeuralNetworkLayers(inputSize, hyperparams_num, fixedNetParams, []);

        % G. Training Options
        opts = createTrainingOptions(hyperparams_num, fixedNetParams, X_val_norm, Y_val_cat);

        % H. Train
        [net, ~] = TrainNetwork(X_train_norm, Y_train_cat, layers, opts);

        % I. Evaluate
        [Y_pred, Y_scores] = classify(net, X_val_norm);

        % Calculate metrics relative to the "Positive" class (Helicopter)
        % Assumes 'Helicopter' is the first class in fixedNetParams
        posClassName = char(fixedNetParams.ClassNames(1));

        fold_metrics = CalculateMetrics(Y_val_cat, Y_pred, Y_scores, posClassName, fixedNetParams.ClassNames);
        inner_metrics_cell{k} = fold_metrics;

        % J. Objective Calculation (1 - F1)
        if isfield(fold_metrics, 'PositiveF1_score') && ~isnan(fold_metrics.PositiveF1_score)
            inner_objectives(k) = 1 - fold_metrics.PositiveAUC_ROC;
        else
            inner_objectives(k) = 1.0; % Penalty for NaN result
        end

    catch ME
        % Robust failure handling: Don't crash the whole optimization
        fprintf('    > Inner Fold %d failed: %s\n', k, ME.message);
        inner_objectives(k) = 1.0; % Max penalty
        inner_metrics_cell{k} = [];
    end
end

% --- Aggregation ---
objective = mean(inner_objectives);
constraints = []; % Constraints handled externally by BayesOpt wrapper

% Return detailed stats for stability analysis
UserData.MeanObjective = objective;
UserData.StdObjective  = std(inner_objectives);
UserData.FoldMetrics   = inner_metrics_cell;

% Garbage collection to manage GPU memory
clear net X_train_norm X_val_norm;
reset(gpuDevice); % Force VRAM cleanup

end