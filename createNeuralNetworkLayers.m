function layers = createNeuralNetworkLayers(numFeatures, hyperparams, fixedNetParams, classWeights)
% CREATENEURALNETWORKLAYERS Creates neural network layer architecture

% Set default empty classWeights if not provided
if nargin < 4
    classWeights = [];
end

% Convert table to struct if needed
if istable(hyperparams)
    hyperparams = table2struct(hyperparams);
end

layers = [
    featureInputLayer(numFeatures, 'Normalization', 'none')
    fullyConnectedLayer(hyperparams.Layer1Size)
    reluLayer
    dropoutLayer(hyperparams.DropoutP)
    fullyConnectedLayer(hyperparams.Layer2Size)
    reluLayer
    dropoutLayer(hyperparams.DropoutP)
    fullyConnectedLayer(numel(fixedNetParams.ClassNames))
    softmaxLayer
    ];

% FIXED: Conditional classificationLayer creation
if isempty(classWeights)
    % Don't pass ClassWeights parameter when empty
    layers = [layers; classificationLayer('Classes', fixedNetParams.ClassNames)];
else
    % Pass ClassWeights when provided
    layers = [layers; classificationLayer('Classes', fixedNetParams.ClassNames, 'ClassWeights', classWeights)];
end

end