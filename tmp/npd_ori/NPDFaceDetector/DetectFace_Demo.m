close all; clear; clc;

modelFile = 'model_frontal.mat';
imgFile = 'rokid.jpg';

load(modelFile, 'npdModel');

img = imread(imgFile);
tic;
rects = DetectFace(npdModel, img);
toc;

numFaces = length(rects);    
fprintf('%d faces detected.\n', numFaces);

if numFaces > 0
    border = round(size(img,2) / 300);
    if border < 2, border = 2; end

    for j = 1 : numFaces
        img = DrawRectangle(img, rects(j).col, rects(j).row, rects(j).width, rects(j).height, [0 255 0], border);
    end
end

%imshow(img);
