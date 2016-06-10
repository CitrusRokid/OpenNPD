% This demo shows how to train an NPD feature based face detector

close all; clear; clc; dbstop if error;

addpath('../bin/');

if ~exist(['NPD.' mexext], 'file')
    Compile;
end

options.objSize = 20; % size of the face detection template
options.negRatio = 1; % factor of bootstrap nonface samples. For example, 
% negRatio=2 means bootstrapping two times of nonface samples w.r.t face samples.
options.finalNegs = 100; % the minimal number of bootstrapped nonface samples. 
% The training will be stopped if there is no enough nonface samples in the
% final stage. This is also to avoid overfitting.
options.numFaces = 100; % the number of face samples to be used for training.
% Inf means to use all face samples.
options.numThreads = 24; % the number of computing threads for bootstrapping

options.boostOpt.treeLevel = 4; % the maximal depth of the DQT trees to be learned
options.boostOpt.maxNumWeaks = 4000; % maximal number of weak classifiers to be learned
options.boostOpt.minDR = 1.0; % minimal detection rate required
options.boostOpt.maxFAR = 1e-16; % maximal FAR allowed; stop the training if reached
options.boostOpt.minSamples = 100; % minimal samples required to continue training. 1000 is preferred in practice
% for other options to control the learning, please see LearnGAB.m.

faceDBFile = '../data/FaceDB.mat';
nonfaceDBFile = '../data/NonfaceDB.mat';
outFile = '../result.mat';

model = TrainDetector(faceDBFile, nonfaceDBFile, outFile, options);

rmpath('../bin/');
