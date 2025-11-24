function metrics = CalculateMetrics(Y_true_cat, Y_pred_labels, Y_pred_scores, positiveClassName, classNamesAll)
% CalculateMetrics Computes classification performance metrics for all classes.
%
% Inputs:
%   Y_true_cat        - True labels (categorical vector).
%   Y_pred_labels     - Predicted labels (categorical vector).
%   Y_pred_scores     - Predicted posterior probabilities (numeric matrix,
%                       samples×numClasses). Columns correspond to classNamesAll.
%   positiveClassName - String/char name of positive class (for backward compatibility).
%   classNamesAll     - Categorical array or cell array of strings of all class names,
%                       in the same order as columns of Y_pred_scores.
%
% Outputs:
%   metrics - Struct containing:
%       .ObjectiveMetric      (1 - Accuracy)
%       .Accuracy             (overall accuracy)
%       .ConfusionMatrix      (numClasses×numClasses)
%       .PerClassPrecision    (1×numClasses)
%       .PerClassRecall       (1×numClasses)
%       .PerClassF1           (1×numClasses)
%       .PerClassSpecificity  (1×numClasses)
%       .PerClassAUC_ROC      (1×numClasses)
%       .PerClassPR_AUC       (1×numClasses)
%       .PositivePrecision    (scalar, for positiveClassName)
%       .PositiveRecall       (scalar)
%       .PositiveF1_score     (scalar)
%       .PositiveSpecificity  (scalar)
%       .PositiveAUC_ROC      (scalar)
%
% Note: positiveClassName must appear in classNamesAll.
%

% Initialize output struct with NaN or empty defaults
metrics = struct();
metrics.ObjectiveMetric     = Inf;
metrics.Accuracy            = NaN;
metrics.ConfusionMatrix     = [];
metrics.PerClassPrecision   = [];
metrics.PerClassRecall      = [];
metrics.PerClassF1          = [];
metrics.PerClassSpecificity = [];
metrics.PerClassAUC_ROC     = [];
metrics.PerClassPR_AUC      = [];
metrics.PositivePrecision   = NaN;
metrics.PositiveRecall      = NaN;
metrics.PositiveF1_score    = NaN;
metrics.PositiveSpecificity = NaN;
metrics.PositiveAUC_ROC     = NaN;
metrics.PositivePR_AUC      = NaN;

% Validate inputs
if isempty(Y_true_cat) || isempty(Y_pred_labels)
    warning('CalculateMetrics: True or predicted labels are empty. Returning empty metrics.');
    return;
end
if ~iscategorical(Y_true_cat)
    Y_true_cat = categorical(Y_true_cat);
end
if ~iscategorical(Y_pred_labels)
    Y_pred_labels = categorical(Y_pred_labels, categories(Y_true_cat));
end
if ~iscategorical(classNamesAll)
    classNamesAll = categorical(classNamesAll);
end

% Ensure consistent category ordering
expectedCats = categories(classNamesAll);
Y_true_cat   = reordercats(Y_true_cat,   expectedCats);
Y_pred_labels= reordercats(Y_pred_labels,expectedCats);

% Compute confusion matrix
C = confusionmat(Y_true_cat, Y_pred_labels, 'Order', expectedCats);
metrics.ConfusionMatrix = C;

numClasses = numel(expectedCats);
positiveIdx = find(strcmp(cellstr(expectedCats), positiveClassName), 1);
if isempty(positiveIdx)
    warning('CalculateMetrics: positiveClassName "%s" not found. Class-specific metrics for positive class will be NaN.', positiveClassName);
end

% Compute overall accuracy
totalObs = sum(C(:));
if totalObs > 0
    acc = sum(diag(C)) / totalObs;
else
    acc = NaN;
end
metrics.Accuracy = acc;
metrics.ObjectiveMetric = 1 - acc;

% Allocate per-class vectors
precVec       = nan(1,numClasses);
recVec        = nan(1,numClasses);
f1Vec         = nan(1,numClasses);
specVec       = nan(1,numClasses);
aucROCVec     = nan(1,numClasses);
prAUCVec      = nan(1,numClasses);

% Loop over each class
for c = 1:numClasses
    % True/false positives/negatives for class c
    TP = C(c,c);
    FN = sum(C(c,:)) - TP;
    FP = sum(C(:,c)) - TP;
    TN = totalObs - (TP + FP + FN);

    % Precision
    if (TP + FP) > 0
        precVec(c) = TP / (TP + FP);
    else
        precVec(c) = 0;
    end

    % Recall (Sensitivity)
    if (TP + FN) > 0
        recVec(c) = TP / (TP + FN);
    else
        recVec(c) = 0;
    end

    % F1 score
    if (precVec(c) + recVec(c)) > 0
        f1Vec(c) = 2 * (precVec(c) * recVec(c)) / (precVec(c) + recVec(c));
    else
        f1Vec(c) = 0;
    end

    % Specificity
    if (TN + FP) > 0
        specVec(c) = TN / (TN + FP);
    else
        specVec(c) = 0;
    end

    % AUC-ROC for this class
    if ~isempty(Y_pred_scores) && size(Y_pred_scores,2) == numClasses && numel(Y_true_cat) == size(Y_pred_scores,1)
        scoresC = Y_pred_scores(:,c);
        uniqueTrue = unique(Y_true_cat);
        if numel(uniqueTrue) > 1
            try
                [~,~,~, aucVal] = perfcurve(Y_true_cat, scoresC, cellstr(expectedCats{c}));
                aucROCVec(c) = aucVal;
            catch
                aucROCVec(c) = NaN;
            end
        else
            aucROCVec(c) = NaN;
        end
    else
        aucROCVec(c) = NaN;
    end

    % PR-AUC for this class
    if ~isempty(Y_pred_scores) && size(Y_pred_scores,2) == numClasses && numel(Y_true_cat) == size(Y_pred_scores,1)
        scoresC = Y_pred_scores(:,c);
        uniqueTrue = unique(Y_true_cat);
        if numel(uniqueTrue) > 1
            try
                [~,~,~, prAucVal] = perfcurve(Y_true_cat, scoresC, cellstr(expectedCats{c}), 'xCrit','reca','yCrit','prec');
                prAUCVec(c) = prAucVal;
            catch
                prAUCVec(c) = NaN;
            end
        else
            prAUCVec(c) = NaN;
        end
    else
        prAUCVec(c) = NaN;
    end
end

% Store per-class results
metrics.PerClassPrecision   = precVec;
metrics.PerClassRecall      = recVec;
metrics.PerClassF1          = f1Vec;
metrics.PerClassSpecificity = specVec;
metrics.PerClassAUC_ROC     = aucROCVec;
metrics.PerClassPR_AUC      = prAUCVec;

% If positive class was found, populate its specific metrics
if ~isempty(positiveIdx)
    metrics.PositivePrecision   = precVec(positiveIdx);
    metrics.PositiveRecall      = recVec(positiveIdx);
    metrics.PositiveF1_score    = f1Vec(positiveIdx);
    metrics.PositiveSpecificity = specVec(positiveIdx);
    metrics.PositiveAUC_ROC     = aucROCVec(positiveIdx);
    metrics.PositivePR_AUC      = prAUCVec(positiveIdx);
end
end
