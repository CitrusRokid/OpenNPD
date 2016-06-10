This MATLAB package provides the Deep Quadratic Tree (DQT) and the Normalized Pixel Difference (NPD) based face detector training method proposed in our PAMI paper [1].

The package contains the following components.

(1) Training codes: CalcTreeDepth.m, LearnGAB.m, PackNPDModel.m, TestDQT.m, TestGAB.m, and TrainDetector.m in the /src directory. These are MATLAB functions for the proposed detector training method.

(2) MEX functions: LearnDQT.cpp, NPD.cpp, NPDClassify.cpp, NPDGrid.cpp, and NPDScan.cpp in the /src directory and the pre-compiled MEX binaries in the /bin directory.

(3) Demo codes: Compile.m and TrainDetector_Demo.m. TrainDetector_Demo.m shows how to train an NPD feature based face detector. Compile.m shows how to compile the mex functions in this package, especially how to enable the -openmp option which is important for speedup. You can follow the Compile.m to compile the mex functions in your own platform.

(4) Sample training data in the /data directory.

How to prepare the training data for this training package?

We use two MAT files to store training data, one for face samples, and the other one for nonface samples. Specifically, the face data file contains a sinle array FaceDB of size [objSize, objSize, numFaces]. The nonface data file contains the following variables:
	numSamples: the number of cropped nonface images of size [objSize, objSize].
	numNonfaceImgs: the number of big nonface images for bootstrapping.
	NonfaceDB: an array of size [objSize, objSize, numSamples] containing the cropped nonface images. This is used in the begining stages of the detector training.
	NonfaceImages: a cell of size [numNonfaceImgs, 1] containing the big nonface images for bootstrapping.


For a quick start, run the TrainDetector_Demo.m code for a demo of the NPD face detector training. It should be finished in several tens of seconds.



Version: 1.0
Date: 2015-10-04
 
Author: Shengcai Liao
Institute: National Laboratory of Pattern Recognition, 	Institute of Automation, Chinese Academy of Sciences

Email: scliao@nlpr.ia.ac.cn

Project page: http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/


References:

[1] Shengcai Liao, Anil K. Jain, and Stan Z. Li, "A Fast and Accurate Unconstrained Face Detector," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015 (Accepted).

