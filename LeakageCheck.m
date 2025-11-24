function [isLeakFree, reportTbl] = LeakageCheck(tbl1, tbl2)
% LEAKAGECHECK Verify no overlap in SourceID and report class balance.
%   [isLeakFree, reportTbl] = LeakageCheck(trainTbl, testTbl)
%
%   Inputs:
%     tbl1, tbl2 - Tables containing 'SourceID' and 'AHM_7_Class'.
%
%   Outputs:
%     isLeakFree - Logical true if sets are disjoint by SourceID.
%     reportTbl  - Summary of rows, Heli counts, and Other counts.

% --- 1. Input Validation ---
for T = {tbl1, tbl2}
    if ~istable(T{1})
        error('LeakageCheck:InvalidInput', 'Inputs must be tables.');
    end
    vars = T{1}.Properties.VariableNames;
    if ~ismember('SourceID', vars)
        error('LeakageCheck:MissingVar', 'Table is missing ''SourceID''.');
    end
    if ~ismember('AHM_7_Class', vars)
        error('LeakageCheck:MissingVar', 'Table is missing ''AHM_7_Class''.');
    end
end

% --- 2. Leakage Verification (SourceID) ---
% Convert to string to ensure robust comparison
src1 = string(tbl1.SourceID);
src2 = string(tbl2.SourceID);

intersectItems = intersect(src1, src2);
isLeakFree = isempty(intersectItems);

if ~isLeakFree
    fprintf(2, 'LeakageCheck: WARNING! Found %d overlapping SourceIDs.\n', numel(intersectItems));
    disp(head(intersectItems)); % Show first few offenders
end

% --- 3. Distribution Stats ---
    function [nHeli, nOther] = count_classes(t)
        lbls = string(t.AHM_7_Class);
        nHeli = sum(lbls == "Helicopter");
        nOther = sum(lbls ~= "Helicopter");
    end

[h1, o1] = count_classes(tbl1);
[h2, o2] = count_classes(tbl2);

% Ratio (Heli / Other)
r1 = h1 / max(1, o1);
r2 = h2 / max(1, o2);

% --- 4. Build Report ---
% Determine variable names from input if possible
n1 = inputname(1); if isempty(n1), n1 = 'Set_1'; end
n2 = inputname(2); if isempty(n2), n2 = 'Set_2'; end

reportTbl = table( ...
    string({n1; n2}), ...
    [height(tbl1); height(tbl2)], ...
    [h1; h2], ...
    [o1; o2], ...
    [r1; r2], ...
    'VariableNames', {'SetName', 'TotalRows', 'Helicopter', 'Others', 'Ratio_H_O'} ...
    );
end