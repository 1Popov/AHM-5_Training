function results_bayes = BayesOpt(objFcn, optVars, numBayesoptEvaluations, foldIdx, xConstraint)
% BayesOpt Wrapper for robust Bayesian Optimization with constraints.
%
% Inputs:
%   objFcn: Handle to the objective function
%   optVars: Array of optimizableVariable objects
%   numBayesoptEvaluations: Total budget for evaluations
%   foldIdx: [OuterFold, InnerFold] for logging context
%   xConstraint: Handle to constraint function (e.g., memory budget)

% Handle optional XConstraintFcn for backward compatibility
if nargin < 5
    xConstraint = [];
end

% Configuration for resource management
% Note: MaxTime is a safety net (32 hours), but Evals usually finish first.
maxTime = 32 * 60 * 60;

% Phase 1 Configuration (Exploration)
minEvals = 25;

% Cap total evals to the user's request (or a safe limit for debug)
maxEvals = numBayesoptEvaluations;

% Convergence monitoring
patience = 2;
improvement_threshold = 0.001;

startTime = tic;

% --- Log Header ---
fprintf('\n--- Outer Fold %d: Bayesian Optimization Log ---\n', foldIdx(1));
if ~isempty(xConstraint)
    fprintf('    (Constraints active: Preventing High-Compute/Low-Batch combinations)\n');
end

% --- Phase 1: Initial Exploration ---
% We run a fixed number of evaluations to map the space before checking convergence.
fprintf('Phase 1: Initial Exploration (%d evals)...\n', minEvals);

results_bayes = bayesopt(objFcn, optVars, ...
    'MaxObjectiveEvaluations', minEvals, ...
    'MaxTime', maxTime, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'XConstraintFcn', xConstraint, ...
    'NumSeedPoints', 4, ... % Increased slightly to help constraint discovery
    'Verbose', 1, ...
    'PlotFcn', [], ...
    'UseParallel', false); % Disable parallel inside bayesopt; handled by outer loop if needed

% --- Stability Analysis (Post-Phase 1) ---
fprintf('Computing optimization stability metrics...\n');
if isfield(results_bayes, 'UserDataTrace') && ~isempty(results_bayes.UserDataTrace)
    userDataTrace = results_bayes.UserDataTrace;

    % Safe extraction helpers
    reliabilityScores = arrayfun(@(x) getFieldSafe(x, 'MetricStability.ReliabilityScore', 0), userDataTrace);
    metricAgreements  = arrayfun(@(x) getFieldSafe(x, 'MetricAgreement', NaN), userDataTrace);
    overfittingGaps   = arrayfun(@(x) getFieldSafe(x, 'MetricStability.OverfittingGap', NaN), userDataTrace);

    stability_metrics = struct();
    stability_metrics.MeanReliability = mean(reliabilityScores(reliabilityScores > 0));
    stability_metrics.MetricConsistency = mean(metricAgreements(~isnan(metricAgreements)));
    stability_metrics.OverfittingFrequency = mean(overfittingGaps > 0.15, 'omitnan');

    results_bayes.StabilityAnalysis = stability_metrics;

    fprintf('  Reliability: %.3f | Consistency: %.3f | Overfitting: %.1f%%\n', ...
        stability_metrics.MeanReliability, stability_metrics.MetricConsistency, ...
        stability_metrics.OverfittingFrequency * 100);
end

% --- Phase 2: Convergence Monitoring ---
bestObjective = min(results_bayes.ObjectiveTrace);
stagnationCount = 0;
currentEvals = length(results_bayes.ObjectiveTrace);

while currentEvals < maxEvals
    % Adaptive batch sizing for Phase 2
    remainingEvals = maxEvals - currentEvals;
    remainingTime = maxTime - toc(startTime);

    if remainingTime < 600 % Less than 10 mins left
        batchSize = min(2, remainingEvals);
    else
        batchSize = min(5, remainingEvals);
    end

    if batchSize <= 0, break; end

    fprintf('Phase 2: Resuming for %d more evaluations (stagnation: %d/%d)...\n', ...
        batchSize, stagnationCount, patience);

    % Resume optimization
    results_bayes = bayesopt(objFcn, optVars, ...
        'MaxObjectiveEvaluations', currentEvals + batchSize, ...
        'MaxTime', maxTime, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'XConstraintFcn', xConstraint, ...
        'InitialX', results_bayes.XTrace, ...
        'InitialObjective', results_bayes.ObjectiveTrace, ...
        'Verbose', 1, ...
        'PlotFcn', [], ...
        'UseParallel', false);

    % Check for improvement
    newEvals = length(results_bayes.ObjectiveTrace);
    newBest = min(results_bayes.ObjectiveTrace);
    improvement = bestObjective - newBest;

    if improvement > improvement_threshold
        bestObjective = newBest;
        stagnationCount = 0;
        fprintf('  > Improved! New Best Objective: %.4f\n', bestObjective);
    else
        stagnationCount = stagnationCount + 1;
        fprintf('  > No significant improvement. Stagnation count: %d\n', stagnationCount);
    end

    currentEvals = newEvals;

    if stagnationCount >= patience
        fprintf('Optimization stopped early due to stagnation after %d evaluations.\n', currentEvals);
        break;
    end
end

fprintf('Fold [%d,%d] Finished: Obj=%.4f, Evals=%d/%d, Time=%.1fm\n', ...
    foldIdx(1), foldIdx(2), bestObjective, currentEvals, maxEvals, toc(startTime)/60);

end

% --- Helper for Robust Field Access ---
function value = getFieldSafe(struct_var, fieldPath, defaultValue)
try
    fields = strsplit(fieldPath, '.');
    value = struct_var;
    for i = 1:length(fields)
        if isfield(value, fields{i})
            value = value.(fields{i});
        else
            value = defaultValue;
            return;
        end
    end
catch
    value = defaultValue;
end
end