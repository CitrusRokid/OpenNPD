function [model, negPassIndex, posFx, negFx] = LearnGAB(posX, negX, model, options)
%% [model, negPassIndex, posFx, negFx] = LearnGAB(posX, negX, model, options)
% Train a soft cascade based Gentle AdaBoost classifier, with deep quadratic 
% tree (DQT) based weak classifiers.
%
% Input:
%   <posX>: features of the positive samples. Size: [nPos, d].
%   <negX>: features of the negative samples. Size: [nNeg, d].
%   <model>: the previously learned model. Leave it empty at beginning.
%   [optioins]: optional parameters. See the beginning codes of this function 
%   for the parameter meanings and the default values.
%
% Output:
%   model: output of the trained detector.
%   negPassIndex: the index of negative samples that passed the classifier.
%   posFx: classifier scores for positive samples.
%   negFx: classifier scores for negative samples.
% 
% Example:
%{
    N = 1000; R1 = 3; R2 = 4;

    X1 = randn(N,2);
    nor = sqrt(sum(X1.^2, 2));
    index = nor > R1;
    X1(index, :) = bsxfun(@rdivide, X1(index, :), nor(index)) * R1;

    X2 = randn(N,2) * 8;
    nor = sqrt(sum(X2.^2, 2));
    index = nor < R2;
    X2(index, :) = bsxfun(@rdivide, X2(index, :), nor(index)) * R2;

    minVal = min(min(X1(:)), min(X2(:)));
    maxVal = max(max(X1(:)), max(X2(:)));
    X1 = uint8( floor((X1 - minVal) / (maxVal - minVal) * 255) );
    X2 = uint8( floor((X2 - minVal) / (maxVal - minVal) * 255) );
    figure; plot(X1(:,1),X1(:,2),'r+',X2(:,1),X2(:,2),'bo');

    opt.treeLevel = 2;
    opt.minLeaf = 1 / 2^6;
    opt.maxNumWeaks = 10;
    opt.maxFAR = 0.01;
    opt.minSamples = 10;
    opt.minNegRatio = 0.01;

    [model, negPassIndex, posFx, negFx] = LearnGAB(X1, X2, [], opt);
    [posFx2, posPass] = TestGAB(model, X1);
    [negFx2, negPass] = TestGAB(model, X2);
%}
%
%  Reference:
%     Shengcai Liao, Anil K. Jain, and Stan Z. Li, "A Fast and Accurate Unconstrained Face Detector," 
%       IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015 (Accepted).
% 
% Version: 1.0
% Date: 2015-10-01
%
% Author: Shengcai Liao
% Institute: National Laboratory of Pattern Recognition,
%   Institute of Automation, Chinese Academy of Sciences
% Email: scliao@nlpr.ia.ac.cn
%
% ----------------------------------
% Copyright (c) 2015 Shengcai Liao
% ----------------------------------

treeLevel = 4; % the maximal depth of the DQT trees to be learned
maxNumWeaks = 1000; % maximal number of weak classifiers to be learned
minDR = 1.0; % minimal detection rate required
maxFAR = 1e-16; % maximal FAR allowed; stop the training if reached
minSamples = 1000; % minimal samples required to continue training
minNegRatio = 0.2; % minimal fraction of negative samples required to remain, 
% w.r.t. the total number of negative samples. This is a signal of
% requiring new negative sample bootstrapping. Also used to avoid
% overfitting.
trimFrac = 0.05; % weight trimming in AdaBoost
samFrac = 1.0; % the fraction of samples randomly selected in each iteration 
% for training; could be used to avoid overfitting.
minLeafFrac = 0.01; % minimal sample fraction w.r.t. the total number of 
% samples required in each leaf node. This is used to avoid overfitting.
minLeaf = 100; % minimal samples required in each leaf node. This is used to avoid overfitting.
maxWeight = 100; % maximal sample weight in AdaBoost; used to ensure numerical stability.
numThreads = 24; % the number of computing threads in tree learning

if nargin >= 4 && ~isempty(options)
    if isfield(options,'maxWeight') && ~isempty(options.maxWeight)
        maxWeight = options.maxWeight;
    end
    if isfield(options,'samFrac') && ~isempty(options.samFrac)
        samFrac = options.samFrac;
    end
    if isfield(options,'treeLevel') && ~isempty(options.treeLevel)
        treeLevel = options.treeLevel;
    end
    if isfield(options,'minLeafFrac') && ~isempty(options.minLeafFrac)
        minLeafFrac = options.minLeafFrac;
    end
    if isfield(options,'minLeaf') && ~isempty(options.minLeaf)
        minLeaf = options.minLeaf;
    end
    if isfield(options,'maxNumWeaks') && ~isempty(options.maxNumWeaks)
        maxNumWeaks = options.maxNumWeaks;
    end
    if isfield(options,'maxFAR') && ~isempty(options.maxFAR)
        maxFAR = options.maxFAR;
    end
    if isfield(options,'minNegRatio') && ~isempty(options.minNegRatio)
        minNegRatio = options.minNegRatio;
    end
    if isfield(options,'minSamples') && ~isempty(options.minSamples)
        minSamples = options.minSamples;
    end
    if isfield(options,'trimFrac') && ~isempty(options.trimFrac)
        trimFrac = options.trimFrac;
    end
    if isfield(options,'numThreads') && ~isempty(options.numThreads)
        numThreads = options.numThreads;
    end
    if isfield(options,'minDR') && ~isempty(options.minDR)
        minDR = options.minDR;
    end
end

nPos = size(posX, 1);
nNeg = size(negX, 1);

t0 = tic;

if nargin >= 3 && ~isempty(model)
    fprintf('Test the current model... ');
    T = length(model);
    
    [posFx, passCount] = TestGAB(model, posX);
    if any(passCount < T)
        warning('Some positive samples cannot pass all stages.');
        index = passCount == T;
        posX = posX(index, :);
        posFx = posFx(index);
        nPos = size(posX, 1);
    end
    
    [negFx, passCount] = TestGAB(model, negX);
    if any(passCount < length(model))
        warning('Some negative samples cannot pass all stages.');
        index = passCount == T;
        negX = negX(index, :);
        negFx = negFx(index);
        nNeg = size(negX, 1);
    end
    
    posW = CalcWeight(posFx, 1, maxWeight);
    negW = CalcWeight(negFx, -1, maxWeight);
    startIter = T + 1;
    
    fprintf('%.3f seconds.\n', toc(t0));
else
    posW = ones(nPos, 1, 'single') / nPos;
    negW = ones(nNeg, 1, 'single') / nNeg;
    posFx = zeros(nPos, 1, 'single');
    negFx = zeros(nNeg, 1, 'single');
    startIter = 1;
end

negPassIndex = 1 : nNeg;
nNegPass = nNeg;

fprintf('Start to train AdaBoost. nPos=%d, nNeg=%d\n\n', nPos, nNeg);


for t = startIter : maxNumWeaks    
    if nNegPass < minSamples
        fprintf('\nNo enough negative samples. The AdaBoost learning terminates at iteration %d. nNegPass = %d.\n', t - 1, nNegPass);
        break;
    end    
    
    nPosSam = max(round(nPos * samFrac), minSamples);
    posIndex = randperm(nPos);
    posIndex = posIndex(1 : nPosSam);
    
    nNegSam = max(round(nNegPass * samFrac), minSamples);
    negIndex = randperm(nNegPass);
    negIndex = negPassIndex( negIndex(1 : nNegSam) );
    
    % trim weight
    w = sort( posW(posIndex) );
    k = find(cumsum(w) >= trimFrac, 1, 'first');
    k = min(k, nPosSam - minSamples + 1);
    posIndex = posIndex( posW(posIndex) >= w(k) );
    
    w = sort( negW(negIndex) );
    k = find(cumsum(w) >= trimFrac, 1, 'first');
    k = min(k, nNegSam - minSamples + 1);
    negIndex = negIndex( negW(negIndex) >= w(k) );
    
    nPosSam = length(posIndex);
    nNegSam = length(negIndex);
    minLeaf_t = max( round((nPosSam + nNegSam) * minLeafFrac), minLeaf);
    
    fprintf('Iter %d: nPos=%d, nNeg=%d, ', t, nPosSam, nNegSam); 
    
    [feaId, cutpoint, leftChild, rightChild, fit, minCost] = LearnDQT(posX, negX, posW, negW, ...
        posFx, negFx, int32(posIndex - 1), int32(negIndex - 1), treeLevel, minLeaf_t, numThreads);
    
    if isempty(feaId)
        fprintf('\n\nNo available features to satisfy the split. The AdaBoost learning terminates.\n');
        break;
    end
    
    model(t).feaId = feaId';
    model(t).cutpoint = cutpoint;
    model(t).leftChild = leftChild;
    model(t).rightChild = rightChild;
    model(t).fit = fit;
    model(t).depth = CalcTreeDepth(leftChild, rightChild);
    
    posFx = posFx + TestDQT(model(t), posX(:, feaId));
    negFx(negPassIndex) = negFx(negPassIndex) + TestDQT(model(t), negX(negPassIndex, feaId));
    
    v = sort(posFx);
    index = max(floor(nPos*(1-minDR)),1);
    model(t).threshold = v(index);
    
    negPassIndex = negPassIndex( negFx(negPassIndex) >= model(t).threshold );
    model(t).far = length(negPassIndex) / nNegPass;
    nNegPass = length(negPassIndex);
    FAR = prod([model.far]);
    
    aveEval = model(1).depth + sum( [model(2:end).depth] .* cumprod([model(1:end-1).far]) );
    
    fprintf('FAR(t)=%.2f%%, FAR=%.2g, depth=%d, nFea(t)=%d, nFea=%d, cost=%.3f.\n', ...
        model(t).far*100, FAR, model(t).depth, length(feaId), length([model.feaId]), minCost);
    fprintf('\t\tnNegPass=%d, aveEval=%.3f, time=%.0fs, meanT=%.3fs.\n', nNegPass, aveEval, toc(t0), toc(t0)/(t-startIter+1));
    
    if FAR <= maxFAR
        fprintf('\n\nThe training is converged at iteration %d. FAR = %.2f%%\n', t, FAR * 100);
        break;
    end
    
    if nNegPass < nNeg * minNegRatio || nNegPass < minSamples
        fprintf('\n\nNo enough negative samples. The AdaBoost learning terminates at iteration %d. nNegPass = %d.\n', t, nNegPass);
        break;
    end
    
    posW = CalcWeight(posFx, 1, maxWeight);    
    negW(negPassIndex) = CalcWeight(negFx(negPassIndex), -1, maxWeight);
end

fprintf('\n\nThe adaboost training is finished. Total time: %.0f seconds. Mean time: %.3f seconds.\n\n', toc(t0), toc(t0)/(t-startIter+1));
end


function weight = CalcWeight(Fx, y, maxWeight)
    n = length(Fx);
    weight = min(exp(-y * Fx), maxWeight);
    s = sum(weight);
    if s == 0
        weight = ones(n, 1) / n;
    else
        weight = weight / s;
    end
end
