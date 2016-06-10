function outModel = PackNPDModel(inModel, objSize, scaleFactor, maxFace)
% function to pack the learned NPD face detector for the use of the mex functions.

if nargin < 4
    maxFace = 4000;
    if nargin < 3
        scaleFactor = 1.2;
    end
end

numStages = length(inModel);
if numStages == 0
    outModel = [];
    return;
end

feaId = {inModel.feaId};
numFeas = cellfun(@length, feaId);
treeRoot = [0, cumsum(numFeas(1:numStages-1))];
feaId = cell2mat(feaId);
numBranchNodes = length(feaId);

fit = {inModel.fit};
numLeaf = cellfun(@length, fit);
leafRoot = [0, cumsum(numLeaf(1:numStages-1))];
fit = cell2mat(fit');
numLeafNodes = length(fit);

stageThreshold = [inModel.threshold];

numPixels = objSize * objSize;
mask = tril(true(numPixels,numPixels),-1);
[points2, points1] = find(mask);
pixel1 = points1(feaId) - 1;
pixel2 = points2(feaId) - 1;

cutpoint = [inModel.cutpoint];

leftChild = cell2mat({inModel.leftChild}');
rightChild = cell2mat({inModel.rightChild}');

for i = 2 : numStages
    branchOffset = treeRoot(i);
    leafOffset = leafRoot(i);
    index = sum(numFeas(1:i-1)) + 1 : sum(numFeas(1:i));
    
    isLeaf = leftChild(index) < 0;
    leftChild(index(isLeaf)) = leftChild(index(isLeaf)) - leafOffset;
    leftChild(index(~isLeaf)) = leftChild(index(~isLeaf)) + branchOffset;
    
    isLeaf = rightChild(index) < 0;
    rightChild(index(isLeaf)) = rightChild(index(isLeaf)) - leafOffset;
    rightChild(index(~isLeaf)) = rightChild(index(~isLeaf)) + branchOffset;
end

[X, Y] = meshgrid(0 : 255, 0 : 255);
npdTable = X ./ (X + Y);
npdTable(1,1) = 1 / 2;
npdTable = min(255, floor(256 * npdTable));

outModel.objSize = int32(objSize);
outModel.numStages = int32(numStages);
outModel.numBranchNodes = int32(numBranchNodes);
outModel.numLeafNodes = int32(numLeafNodes);
outModel.stageThreshold = single(stageThreshold)';

outModel.treeRoot = int32(treeRoot)';
outModel.pixel1 = int32(pixel1);
outModel.pixel2 = int32(pixel2);
outModel.cutpoint = uint8(cutpoint)';
outModel.leftChild = int32(leftChild);
outModel.rightChild = int32(rightChild);
outModel.fit = single(fit);

outModel.npdTable = uint8(npdTable);

aveEval = inModel(1).depth + sum( [inModel(2:end).depth] .* cumprod([inModel(1:end-1).far]) );

fprintf('#weaks: %d\n', numStages);
fprintf('#features: %d\n', numBranchNodes);
fprintf('#average fea evals: %f\n', aveEval);

if scaleFactor > 1
    numScales = floor( log(maxFace / objSize) / log(scaleFactor) ) + 1;
    scale = scaleFactor.^(0:numScales-1);
    winSize = round(objSize .* scale);
    
    points1_row = mod(points1 - 1, objSize);
    points1_col = floor((points1 - 1) / objSize);
    points2_row = mod(points2 - 1, objSize);
    points2_col = floor((points2 - 1) / objSize);

    pixel1 = repmat(pixel1, [1, numScales]);
    pixel2 = repmat(pixel2, [1, numScales]);
    
    for i = 2 : numScales
        points1 = round(points1_col * scale(i)) * winSize(i) + round(points1_row * scale(i));
        points2 = round(points2_col * scale(i)) * winSize(i) + round(points2_row * scale(i));
        pixel1(:, i) = points1(feaId);
        pixel2(:, i) = points2(feaId);
    end
    
    outModel.scaleFactor = double(scaleFactor);
    outModel.numScales = int32(numScales);
    outModel.winSize = int32(winSize);
    outModel.pixel1 = int32(pixel1);
    outModel.pixel2 = int32(pixel2);
end
