function [X_bal, Y_bal] = ClassBalance(X, Y, targetCount)
% ClassBalance: Balances dataset.
% If targetCount is provided: Balances all classes to exactly that number 
% (undersampling majority, oversampling minority).
% If targetCount is omitted: Balances to the size of the largest class.

    Y = string(Y); 
    classes = unique(Y);
    counts = countcats(categorical(Y));
    
    if nargin < 3
        targetCount = max(counts);
    end
    
    X_bal = [];
    Y_bal = [];
    
    for i = 1:length(classes)
        cls = classes(i);
        idx = find(Y == cls);
        numCurrent = length(idx);
        
        if numCurrent > targetCount
            % Undersample (Downsample)
            keepIdx = idx(randperm(numCurrent, targetCount));
        else
            % Oversample (Upsample)
            keepIdx = idx;
            numNeeded = targetCount - numCurrent;
            if numNeeded > 0
                rand_extras = idx(randi(numCurrent, numNeeded, 1));
                keepIdx = [keepIdx; rand_extras];
            end
        end
        
        X_bal = [X_bal; X(keepIdx, :)];
        Y_bal = [Y_bal; Y(keepIdx)];
    end
    
    % Shuffle the balanced dataset
    perm = randperm(length(Y_bal));
    X_bal = X_bal(perm, :);
    Y_bal = Y_bal(perm);
end