function opts = createTrainingOptions(hyperparams, fixedNetParams, X_val, Y_val)
% CREATETRAININGOPTIONS Creates training options for neural network
%
% Inputs:
% hyperparams - Table/struct with InitialLearnRate, Lambda
% fixedNetParams - Struct with fixed training parameters
% X_val, Y_val - Validation data (optional, can be empty)
%
% Output:
% opts - TrainingOptions object

% Convert table to struct if needed
if istable(hyperparams)
 hyperparams = table2struct(hyperparams);
end

% Convert categorical MiniBatchSize to numeric
if isfield(hyperparams, 'MiniBatchSize') && ...
   (iscategorical(hyperparams.MiniBatchSize) || iscell(hyperparams.MiniBatchSize) || isstring(hyperparams.MiniBatchSize))
    hyperparams.MiniBatchSize = str2double(string(hyperparams.MiniBatchSize));
end

% Determine OutputNetwork strategy based on validation data availability
if nargin >= 3 && ~isempty(X_val) && ~isempty(Y_val)
% Use validation-based early stopping
 outputNetwork = 'best-validation-loss';
 validationFreq = fixedNetParams.ValidationFrequency;
 validationPatience = fixedNetParams.ValidationPatience;
 validationData = {X_val, Y_val};
else
% No validation data - use last iteration
 outputNetwork = 'last-iteration';
 validationFreq = [];
 validationPatience = [];
 validationData = {};
end

% Base training options
baseOptions = {'MaxEpochs', fixedNetParams.MaxEpochs, ...
'MiniBatchSize', hyperparams.MiniBatchSize, ...
'InitialLearnRate', hyperparams.InitialLearnRate, ...
'L2Regularization', hyperparams.Lambda, ...
'Shuffle', 'every-epoch', ...
'Verbose', false, ...
'Plots', 'none', ...
'ExecutionEnvironment', fixedNetParams.ExecutionEnvironment, ...
'OutputNetwork', outputNetwork};

% Add new training options if they exist in fixedNetParams
if isfield(fixedNetParams, 'LearnRateSchedule')
 baseOptions = [baseOptions, {'LearnRateSchedule', fixedNetParams.LearnRateSchedule}];
end
if isfield(fixedNetParams, 'LearnRateDropFactor')
 baseOptions = [baseOptions, {'LearnRateDropFactor', fixedNetParams.LearnRateDropFactor}];
end
if isfield(fixedNetParams, 'LearnRateDropPeriod')
 baseOptions = [baseOptions, {'LearnRateDropPeriod', fixedNetParams.LearnRateDropPeriod}];
end
if isfield(fixedNetParams, 'GradientThreshold')
 baseOptions = [baseOptions, {'GradientThreshold', fixedNetParams.GradientThreshold}];
end
if isfield(fixedNetParams, 'GradientThresholdMethod')
 baseOptions = [baseOptions, {'GradientThresholdMethod', fixedNetParams.GradientThresholdMethod}];
end

% Add validation settings if available
if ~isempty(validationData)
 baseOptions = [baseOptions, {'ValidationData', validationData, ...
'ValidationFrequency', validationFreq, ...
'ValidationPatience', validationPatience}];
end

% Create training options with all parameters
opts = trainingOptions(fixedNetParams.Solver, baseOptions{:});
end