function rects = DetectFace(model, I, options)
%% function rects = DetectFace(model, I, options)
% Face detection function for the NPD method
%
% Input:
%   <model>: the learned model for the NPD face detector
%   <I>: the input image to be detected
%   [optioins]: optional parameters. A structure containing any of the
%   following fields:
%       minFace: minimal size of the face that you are searching for. Default: 20
%       maxFace: maximal size of the face that you are searching for. Default: 4000
%       overlappingThreshold: overlapping threshold for grouping nearby
%       detections. Default: 0.5
%       numThreads: number of cpu threads for parallel computing. Default: 24
%
% Output:
%   rects: the detected face retangles.
% 
% Example:
%     I = imread('lena.jpg');
%     load('model.mat');
%     rects = DetectFace(npdModel, I);
%
%  Reference:
%     Shengcai Liao, Anil K. Jain, and Stan Z. Li, "A Fast and Accurate Unconstrained Face Detector," 
%       IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015 (Accepted).
% 
% Version: 1.0
% Date: 2015-03-04
%
% Author: Shengcai Liao
% Institute: National Laboratory of Pattern Recognition,
%   Institute of Automation, Chinese Academy of Sciences
% Email: scliao@nlpr.ia.ac.cn
%
% ----------------------------------
% Copyright (c) 2015 Shengcai Liao
% ----------------------------------

%% initialize parameters
minFace = 20; % minimal size of the face that you are searching for 
maxFace = 4000; % maximal size of the face that you are searching for 
overlappingThreshold = 0.5; % overlapping threshold for grouping nearby detections
numThreads = 24; % number of cpu threads for parallel computing

if nargin > 2 && ~isempty(options)
    if isfield(options, 'minFace') && ~isempty(options.minFace)
        minFace = options.minFace;
    end
    if isfield(options, 'maxFace') && ~isempty(options.maxFace)
        maxFace = options.maxFace;
    end
    if isfield(options, 'overlappingThreshold') && ~isempty(options.overlappingThreshold)
        overlappingThreshold = options.overlappingThreshold;
    end
    if isfield(options, 'numThreads') && ~isempty(options.numThreads)
        numThreads = options.numThreads;
    end
end

%% test detector

if ~ismatrix(I)
    I = rgb2gray(I);
end

candi_rects = NPDScan(model, I, minFace, maxFace, numThreads);

%% post processing
if isempty(candi_rects)
    rects = [];
    return;
end

numCandidates = length(candi_rects);
predicate = eye(numCandidates); % i and j belong to the same group if predicate(i,j) = 1

% mark nearby detections
for i = 1 : numCandidates
    for j = i + 1 : numCandidates
        h = min(candi_rects(i).row + candi_rects(i).size, candi_rects(j).row + candi_rects(j).size) - max(candi_rects(i).row, candi_rects(j).row);
        w = min(candi_rects(i).col + candi_rects(i).size, candi_rects(j).col + candi_rects(j).size) - max(candi_rects(i).col, candi_rects(j).col);
        s = max(h,0) * max(w,0);
        
        if s / (candi_rects(i).size^2 + candi_rects(j).size^2 - s) >= overlappingThreshold
            predicate(i,j) = 1;
            predicate(j,i) = 1;
        end
    end
end

% merge nearby detections
[label, numCandidates] = Partition(predicate);

rects = struct('row', zeros(numCandidates,1), 'col', zeros(numCandidates,1), ...
    'size', zeros(numCandidates,1), 'score', zeros(numCandidates,1), ...
    'neighbors', zeros(numCandidates,1));

for i = 1 : numCandidates
    index = find(label == i);
    weight = Logistic([candi_rects(index).score]');
    rects(i).score = sum( weight );
    rects(i).neighbors = length(index);
    
    if sum(weight) == 0
        weight = ones(length(weight), 1) / length(weight);
    else
        weight = weight / sum(weight);
    end

    rects(i).size = floor([candi_rects(index).size] * weight);
    rects(i).col = floor(([candi_rects(index).col] + [candi_rects(index).size]/2) * weight - rects(i).size/2);
    rects(i).row = floor(([candi_rects(index).row] + [candi_rects(index).size]/2) * weight - rects(i).size/2);
end

clear candi_rects;

% find embeded rectangles
predicate = false(numCandidates); % rect j contains rect i if predicate(i,j) = 1

for i = 1 : numCandidates
    for j = i + 1 : numCandidates
        h = min(rects(i).row + rects(i).size, rects(j).row + rects(j).size) - max(rects(i).row, rects(j).row);
        w = min(rects(i).col + rects(i).size, rects(j).col + rects(j).size) - max(rects(i).col, rects(j).col);
        s = max(h,0) * max(w,0);

        if s / rects(i).size^2 >= overlappingThreshold || s / rects(j).size^2 >= overlappingThreshold
            predicate(i,j) = true;
            predicate(j,i) = true;
        end
    end
end

flag = true(numCandidates,1);

% merge embeded rectangles
for i = 1 : numCandidates
    index = find(predicate(:,i));

    if isempty(index)
        continue;
    end

    s = max([rects(index).score]);
    if s > rects(i).score
        flag(i) = false;
    end
end

rects = rects(flag);

% check borders
[height, width, ~] = size(I);
numFaces = length(rects);

for i = 1 : numFaces
    if rects(i).row < 1
        rects(i).row = 1;
    end
    
    if rects(i).col < 1
        rects(i).col = 1;
    end
    
    rects(i).height = rects(i).size;
    rects(i).width = rects(i).size;
    
    if rects(i).row + rects(i).height - 1 > height
        rects(i).height = height - rects(i).row + 1;
    end
    
    if rects(i).col + rects(i).width - 1 > width
        rects(i).width = width - rects(i).col + 1;
    end    
end
end


function Y = Logistic(X)
    Y = log(1 + exp(X));
    Y(isinf(Y)) = X(isinf(Y));
end
