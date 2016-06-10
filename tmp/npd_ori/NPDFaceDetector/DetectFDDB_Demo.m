close all; clear; clc;

dataDir = 'FDDB/';
imgDir = [dataDir, 'image/original/'];
listFile = [dataDir, 'list/FDDB-list.txt'];
annoFile = [dataDir, 'list/FDDB-ellipseList.txt'];
evalProg = 'evaluation_elps.exe';

modelFile = 'model_unconstrain.mat';
resultFile = 'results.txt';

load(modelFile, 'npdModel');

options.scaleFactor = 1.2;
options.minFace = npdModel.objSize;
options.maxFace = 4000;
options.overlappingThreshold = 0.5;
options.enDelta = 0.1;
options.numThreads = 8;

assert(options.scaleFactor == npdModel.scaleFactor);

[Nim,fileList] = ReadList(listFile);
rects = cell(Nim,1);
fout = fopen(resultFile, 'wt');
T = 0;

for i = 1 : Nim
    imgFile = [imgDir, fileList{i}, '.jpg'];
    img = imread(imgFile);

    tic;
    rects{i} = DetectFace(npdModel, img, options);
    T = T + toc;

    numFaces = length(rects{i});
    fprintf('%d / %d: %d faces detected in "%s"\n', i, Nim, numFaces, fileList{i});
    
    [imgHeight, imgWidth, ~] = size(img);

    for j = 1 : numFaces 
        delta = floor(rects{i}(j).size * options.enDelta);
        r0 = max(rects{i}(j).row - floor(2.5 * delta), 1);
        r1 = min(rects{i}(j).row + rects{i}(j).size - 1 + floor(2.5 * delta), imgHeight);
        c0 = max(rects{i}(j).col - delta, 1);
        c1 = min(rects{i}(j).col + rects{i}(j).size - 1 + delta, imgWidth);

        rects{i}(j).row = r0;
        rects{i}(j).col = c0;
        rects{i}(j).width = c1 - c0 + 1;
        rects{i}(j).height = r1 - r0 + 1;
    end

    fprintf(fout, '%s\n%d\n', fileList{i}, numFaces);

    for j = 1 : numFaces
        fprintf(fout, '%d %d %d %d %f\n', rects{i}(j).col-1, rects{i}(j).row-1, rects{i}(j).width, rects{i}(j).height, rects{i}(j).score);
    end
end

fclose(fout);

cmdStr = sprintf('%s -l "%s" -a "%s" -i "%s" -d "%s" -z ".jpg"', ...
    evalProg, listFile, annoFile, imgDir, resultFile);
system(cmdStr);

fprintf('\nAverage detection time per image: %f second\n', T / Nim);

a = dlmread('tempDiscROC.txt');
a = a(end:-1:1,:);
[~, idx] = unique(a(:,2), 'legacy');
discFP = a(idx,2);
discDR = a(idx,1);
semilogx(discFP, discDR, 'r', 'LineWidth', 2.5); hold on;

a = dlmread('tempContROC.txt');
a = a(end:-1:1,:);
[~, idx] = unique(a(:,2), 'legacy');
contFP = a(idx,2);
contDR = a(idx,1);
semilogx(contFP, contDR, 'm', 'LineWidth', 2.5);

legend('NPD-disc', 'NPD-cont', 'Location', 'SouthEast');
xlim([1,10000]);
grid on;
hold off;
