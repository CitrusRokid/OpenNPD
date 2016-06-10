This MATLAB package provides the NPD face detector proposed in our PAMI paper [1].

The package contains the following components.

(1) Detector codes: NPDScan.cpp, NPDSCan.mexw64, DetectFace.m, Partition.m, DrawRectangle.m. These are MATLAB codes for the proposed detector.

(2) Demo codes: DetectFace_Demo.m and DetectFDDB_Demo.m. These are two demos showing how to use the proposed detector. A sample image lena.jpg, and an auxiliary function ReadList.m are also provided for the evaluation purpose.

(3) Learned detector models: model_frontal.mat, model_unconstrain.mat. model_unconstrain.mat contains the model described in Sec. 4.2.1 in [1], while model_frontal.mat contains the model described in the paragraph above Sec. 5.5 in [1], and also the last column of Table 2 in [1].

For a quick start, run the DetectFace_Demo.m code for a demo of detecting a single image.



Version: 1.0
Date: 2015-08-06
 
Author: Shengcai Liao
Institute: National Laboratory of Pattern Recognition, 	Institute of Automation, Chinese Academy of Sciences

Email: scliao@nlpr.ia.ac.cn

Project page: http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/


References:

[1] Shengcai Liao, Anil K. Jain, and Stan Z. Li, "A Fast and Accurate Unconstrained Face Detector," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015 (Accepted).

