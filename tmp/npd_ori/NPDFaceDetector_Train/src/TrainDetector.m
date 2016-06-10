function model = TrainDetector(faceDBFile, nonfaceDBFile, outFile, options)
%% function model = TrainDetector(faceDBFile, nonfaceDBFile, outFile, options)
% Train a Nomalized Pixel Difference (NPD) based face detector.
%
% Input:
%   <faceDBFile>: MAT file for the face images. It contains an array FaceDB
%   of size [objSize, objSize, numFaces].
%   <nonfaceDBFile>: MAT file for the nonface images.It contains the
%   following variables:
%       numSamples: the number of cropped nonface images of size [objSize,
%       objSize].
%       numNonfaceImgs: the number of big nonface images for bootstrapping.
%       NonfaceDB: an array of size [objSize, objSize, numSamples] 
%           containing the cropped nonface images. This is used in the 
%           begining stages of the detector training.
%       NonfaceImages: a cell of size [numNonfaceImgs, 1] containing the
%       big nonface images for bootstrapping.
%   <outFile>: the output file to store the training result.
%   [optioins]: optional parameters. See the beginning codes of this function
%    for the parameter meanings and the default values.
%
% Output:
%   model: output of the trained detector.
% 
% Example:
%     See TrainDetector_Demo.m.
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

objSize = 20; % size of the face detection template
numFaces = Inf; % the number of face samples to be used for training.
% Inf means to use all face samples.
negRatio = 1; % factor of bootstrap nonface samples. For example, negRatio=2 
% means bootstrapping two times of nonface samples w.r.t face samples.
finalNegs = 1000; % the minimal number of bootstrapped nonface samples. 
% The training will be stopped if there is no enough nonface samples in the
% final stage. This is also to avoid overfitting.
numThreads = 24; % the number of computing threads for bootstrapping

if nargin >= 3 && ~isempty(options)
    if isfield(options,'objSize') && ~isempty(options.objSize)
        objSize = options.objSize;
    end
    if isfield(options,'numFaces') && ~isempty(options.numFaces)
        numFaces = options.numFaces;
    end
    if isfield(options,'negRatio') && ~isempty(options.negRatio)
        negRatio = options.negRatio;
    end
    if isfield(options,'finalNegs') && ~isempty(options.finalNegs)
        finalNegs = options.finalNegs;
    end
    if isfield(options,'numThreads') && ~isempty(options.numThreads)
        numThreads = options.numThreads;
    end
    if isfield(options,'boostOpt') && ~isempty(options.boostOpt)
        boostOpt = options.boostOpt;
    end
end

boostOpt.numThreads = numThreads;

load(faceDBFile, 'FaceDB');
load(nonfaceDBFile, 'numNonfaceImgs', 'numSamples');

numFaces = min(numFaces, size(FaceDB, 3)); %#ok<NODEF>
if numFaces < size(FaceDB, 3)
    index = randperm(size(FaceDB, 3));
    FaceDB = FaceDB(:,:,index(1:numFaces));
end

numNegs = ceil(numFaces * negRatio);

fprintf('Extract face features\n');
faceFea = NPD(FaceDB);
clear FaceDB;

if exist(outFile, 'file')
    load(outFile, 'model', 'npdModel', 'trainTime', 'numStages', 'negIndex', 'numGridFace', 'numSlideFace');
else
    trainTime = 0;
    numStages = 0;
    model = [];
    npdModel = [];
    negIndex = 1 : numSamples;
    numGridFace = ( rand(numNonfaceImgs, 1) + 1 ) * 1e6;
    numSlideFace = ( rand(numNonfaceImgs, 1) + 1 ) * 1e6;
end

NonfaceDB = [];
NonfaceImages = [];

fprintf('Start to train detector.\n');
T = length(model);

while true
    t0 = tic;
    
    [nonfaceFea, NonfaceDB, NonfaceImages, negIndex, numGridFace, numSlideFace] = ...
        BootstrapNonfaces(npdModel, nonfaceDBFile, NonfaceDB, NonfaceImages, objSize, numNegs, negIndex, numGridFace, numSlideFace, numThreads);
    
    if length(negIndex) < 10000
        negIndex = [];
        NonfaceDB = [];
    end
    
    if size(nonfaceFea,1) < finalNegs
        fprintf('\n\nNo enough negative examples to bootstrap (nNeg=%d). The detector training is terminated.\n', size(nonfaceFea,1));
        trainTime = trainTime + toc(t0);
        fprintf('\nTraining time: %.0fs.\n', trainTime);
        break;
    end
    
    if size(nonfaceFea,1) == numNegs
        model = LearnGAB(faceFea, nonfaceFea, model, boostOpt);
    else
        NonfaceDB = [];
        NonfaceImages = [];
        boostOpt2 = boostOpt;
        boostOpt2.minNegRatio = finalNegs / size(nonfaceFea,1);
        model = LearnGAB(faceFea, nonfaceFea, model, boostOpt2);
    end
    
    clear nonfaceFea
    
    npdModel = PackNPDModel(model, objSize);
    
    if length(model) == T
        fprintf('\n\nNo effective features for further detector learning.\n');
        break;
    end
    
    T = length(model);
    
    numStages = numStages + 1;
    trainTime = trainTime + toc(t0);

    try
        save(outFile, 'model', 'npdModel', 'objSize', 'options', 'negRatio', 'numStages', 'trainTime', 'negIndex', 'numGridFace', 'numSlideFace', '-v7.3');
    catch exception
        fprintf('%s\n', exception.message);
        filename = userpath;
        filename = [filename(1:end-1) '\npd_model.mat'];
        fprintf('Save the results in %s instead.\n', filename);
        save(filename, 'model', 'npdModel', 'objSize', 'options', 'negRatio', 'numStages', 'trainTime', 'negIndex', 'numGridFace', 'numSlideFace', '-v7.3');
    end

    far = prod([model.far]);
    fprintf('\nStage %d, #Weaks: %d, FAR: %.2g, Training time: %.0fs, Time per stage: %.0fs, Time per weak: %.3fs.\n\n', ...
        numStages, T, far, trainTime, trainTime / numStages, trainTime / T);
    
    if far <= boostOpt.maxFAR || T == boostOpt.maxNumWeaks || exist('boostOpt2', 'var')
        fprintf('\n\nThe detector training is finished.\n');
        break;
    end
end

clear faceFea nonfaceFea NonfaceDB NonfaceImages

try
    save(outFile, '-v7.3');
catch exception
    fprintf('%s\n', exception.message);
    filename = userpath;
    filename = [filename(1:end-1) '\npd_model.mat'];
    fprintf('Save the results in %s instead.\n', filename);
    save(filename, '-v7.3');
end
end


function [nonfaceFea, NonfaceDB, NonfaceImages, negIndex, numGridFace, numSlideFace, nonfacePatches] = BootstrapNonfaces(npdModel, ...
    nonfaceDBFile, NonfaceDB, NonfaceImages, objSize, numNegs, negIndex, numGridFace, numSlideFace, numThreads)

numLimit = floor(numNegs / 1000);
dispStep = floor(numNegs / 10);
dispCount = 0;

if ~isempty(negIndex) && isempty(NonfaceDB)
    fprintf('Load NonfaceDB... ');
    t1 = tic;
    load(nonfaceDBFile, 'NonfaceDB');
    fprintf('done. %.0f seconds.\n', toc(t1));
end

if isempty(npdModel)
    numNonfaces = size(NonfaceDB, 3);
    index = randperm(numNonfaces);
    nonfacePatches = NonfaceDB(:,:,index(1:numNegs));
else
    nonfacePatches = zeros(objSize, objSize, numNegs, 'uint8');
    T = npdModel.numStages;
    t0 = tic;
    
    if ~isempty(negIndex) && ~isempty(NonfaceDB)
        numValid = length(negIndex);
        [~, passCount] = NPDClassify(npdModel, NonfaceDB, int32(negIndex), numThreads);
        negIndex = negIndex(passCount == T);
        n = length(negIndex);
        
        fprintf('+%g of %.3g NonfaceDB samples. total: %d of %d. Time: %.0f seconds.\n', n, numValid, min(n,numNegs), numNegs, toc(t0));
        index = negIndex;
        
        if n > numNegs
            idx = randperm(n);
            index = index( idx(1:numNegs) );
            n = numNegs;
        end
        
        nonfacePatches(:,:,1:n) = NonfaceDB(:,:,index);
    else
        n = 0;
    end

    if n < numNegs
        if isempty(NonfaceImages)
            NonfaceDB = [];
            negIndex = [];
            
            fprintf('Load NonfaceImages... ');
            t1 = tic;
            load(nonfaceDBFile, 'NonfaceImages');
            fprintf('done. %.0f seconds.\n', toc(t1));
        end
        
        samIndex = find(numGridFace > 0);
        [~, idx] = sort(numGridFace(samIndex), 'descend');
        samIndex = samIndex(idx);
        
        fprintf('%.3g grid samples remained.\n', sum(numGridFace));

        for i = 1 : length(samIndex)
            rects = NPDGrid(npdModel, NonfaceImages{samIndex(i)}, objSize, 4000, numThreads);
            k = length(rects);                
            numGridFace(samIndex(i)) = k;

            if k == 0
                continue;
            end

            if k > numLimit || k > numNegs - n
                index = randperm(k);
                k = min(numLimit, numNegs - n);
                rects = rects(index(1:k));
            end

            for j = 1 : k
                n = n + 1;
                factor = npdModel.scaleFactor^(find(npdModel.winSize == rects(j).size) - 1);
                pIdx = round((0 : objSize - 1) * factor);
                nonfacePatches(:,:,n) = NonfaceImages{samIndex(i)}(rects(j).row + pIdx, rects(j).col + pIdx );
            end
            
            if n > dispStep * (dispCount + 1)
                fprintf('+%d grid samples. total: %d of %d. Time: %.0f seconds.\n', k, n, numNegs, toc(t0));
                dispCount = dispCount + 1;
            end
            
            if n == numNegs
                break;
            end
        end

        if n < numNegs
            samIndex = find(numSlideFace > 0);
            [~, idx] = sort(numSlideFace(samIndex), 'descend');
            samIndex = samIndex(idx);
        
            fprintf('%.3g sliding samples remained.\n', sum(numSlideFace));

            for i = 1 : length(samIndex)
                rects = NPDScan(npdModel, NonfaceImages{samIndex(i)}, objSize, 4000, numThreads);
                k = length(rects);                
                numSlideFace(samIndex(i)) = k;

                if k == 0
                    continue;
                end

                if k > numLimit || k > numNegs - n
                    index = randperm(k);
                    k = min(numLimit, numNegs - n);
                    rects = rects(index(1:k));
                end

                for j = 1 : k
                    n = n + 1;
                    factor = npdModel.scaleFactor^(find(npdModel.winSize == rects(j).size) - 1);
                    pIdx = round((0 : objSize - 1) * factor);
                    nonfacePatches(:,:,n) = NonfaceImages{samIndex(i)}(rects(j).row + pIdx, rects(j).col + pIdx );
                end
            
                if n > dispStep * (dispCount + 1)
                    fprintf('+%d limit-sliding samples. total: %d of %d. Time: %.0f seconds.\n', k, n, numNegs, toc(t0));
                    dispCount = dispCount + 1;
                end

                if n == numNegs
                    break;
                end
            end
        end

        if n < numNegs
            samIndex = find(numSlideFace > numLimit);
            [~, idx] = sort(numSlideFace(samIndex));
            samIndex = samIndex(idx);
        
            fprintf('%.3g sliding samples remained.\n', sum(numSlideFace));

            for i = 1 : length(samIndex)
                rects = NPDScan(npdModel, NonfaceImages{samIndex(i)}, objSize, 4000, numThreads);
                k = length(rects);
                numSlideFace(samIndex(i)) = k;

                if k == 0
                    continue;
                end

                if k > numNegs - n
                    index = randperm(k);
                    k = numNegs - n;
                    rects = rects(index(1:k));
                end

                for j = 1 : k
                    n = n + 1;
                    factor = npdModel.scaleFactor^(find(npdModel.winSize == rects(j).size) - 1);
                    pIdx = round((0 : objSize - 1) * factor);
                    nonfacePatches(:,:,n) = NonfaceImages{samIndex(i)}(rects(j).row + pIdx, rects(j).col + pIdx );
                end
            
                if n > dispStep * (dispCount + 1)
                    fprintf('+%d sliding samples. total: %d of %d. Time: %.0f seconds.\n', k, n, numNegs, toc(t0));
                    dispCount = dispCount + 1;
                end

                if n == numNegs
                    break;
                end
            end
        end

        if n < numNegs
            nonfacePatches(:,:,n+1:end) = [];
        end
    end
end

fprintf('Extract nonface features... ');
t1 = tic;
nonfaceFea = NPD(nonfacePatches);
fprintf('done. %.0f seconds.\n', toc(t1));
end
