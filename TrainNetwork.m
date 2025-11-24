function [trainedNet, trainInfo] = TrainNetwork(X_train_processed, Y_train_cat_processed, layers, trainingAlgoOptions)
% TrainNetwork Trains a neural network using the specified data, layers, and options.
%
% Inputs:
%   X_train_processed     - Training features (N×D numeric matrix, potentially on GPU).
%   Y_train_cat_processed - Training labels (categorical vector, potentially on GPU).
%   layers                - Network definition (array of layer objects).
%   trainingAlgoOptions   - Options struct from trainingOptions().
%
% Outputs:
%   trainedNet            - Trained network object (DAGNetwork / SeriesNetwork).
%   trainInfo             - Struct with training history (loss, accuracy, etc.).

% 1. Input Validation
if nargin < 4
    error('TrainNetwork: Requires data, labels, layers, and options.');
end
if isempty(X_train_processed) || isempty(Y_train_cat_processed)
    error('TrainNetwork: Training data or labels are empty.');
end
if ~isa(layers, 'nnet.cnn.layer.Layer') && ~isa(layers, 'nnet.dlnetwork.LayerGraph')
    warning('TrainNetwork: ''layers'' is not a Layer array or LayerGraph. Ensure it is valid.');
end
if ~isa(trainingAlgoOptions, 'nnet.cnn.TrainingOptions')
    error('TrainNetwork: ''trainingAlgoOptions'' must be a valid TrainingOptions object.');
end

% 2. Train the Neural Network
[trainedNet, trainInfo] = trainNetwork(X_train_processed, Y_train_cat_processed, layers, trainingAlgoOptions);

% Move net and info back to the CPU:
trainedNet = gather(trainedNet);
trainInfo    = gather(trainInfo);

end
