/* Mex function for testing subwindows by the NPD feature based face detector
 *
 * [score, passCount] = NPDClassify(model, patches [, index, numThreads])
 * 
 * Input:
 * 
 *   <model>: structure array of the learned detector.
 *   <patches>: the original input images/patches. Size: [height, width, numImgs].
 *   [index]: indexes of the patches that need to be classified.
 *   [numThreads]: the number of computing threads. Positive double scalar. Default: the number of all available processors.
 *
 * Output:
 *
 *   score: the classifier scores of each (indexed) subwindow.
 *   passCount: the number of stages (weak classifiers) that each (indexed) subwindow passes.
 *
 * Compile:
 *     Windows: mex -largeArrayDims -DWINDOWS 'COMPFLAGS=/openmp $COMPFLAGS' NPDClassify.cpp -v
 *     Linux:  mex -largeArrayDims -DLINUX 'CFLAGS=\$CFLAGS -fopenmp' 'LDFLAGS=\$LDFLAGS -fopenmp' NPDClassify.cpp -v
 * 
 * Reference:
 *     Shengcai Liao, Anil K. Jain, and Stan Z. Li, "A Fast and Accurate Unconstrained Face Detector," 
 *       IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015 (Accepted).
 *
 * Version: 1.0
 * Date: 2015-10-04
 *
 * Author: Shengcai Liao
 * Institute: National Laboratory of Pattern Recognition,
 *   Institute of Automation, Chinese Academy of Sciences
 * Email: scliao@nlpr.ia.ac.cn
 *
 * ----------------------------------
 * Copyright (c) 2015 Shengcai Liao
 * ----------------------------------
*/

#include <omp.h>
#include <math.h>
#include <vector>

using namespace std;

#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{    
    if(nlhs > 2 || nrhs < 2 || nrhs > 4)
    {
        mexErrMsgTxt("Usage: [score, passCount] = NPDClassify(model, patches [, index, numThreads])");
    }
    
    // Set the number of threads
    int numProcs = omp_get_num_procs();
    int numThreads = numProcs;
    
    if(nrhs >= 4 && mxGetScalar(prhs[3]) > 0) numThreads = (int) mxGetScalar(prhs[3]);
    
    if(numThreads > numProcs) numThreads = numProcs;
    omp_set_num_threads(numThreads);
    
    // get input pointers
    const mxArray *pModel = prhs[0];
    
    // get the NPD detector
    int objSize = (int) mxGetScalar(mxGetField(pModel, 0, "objSize"));
    int numStages = (int) mxGetScalar(mxGetField(pModel, 0, "numStages"));
    int numBranchNodes = (int) mxGetScalar(mxGetField(pModel, 0, "numBranchNodes"));
    const float *pStageThreshold = (float *) mxGetData(mxGetField(pModel, 0, "stageThreshold"));
    const int *pTreeRoot = (int *) mxGetData(mxGetField(pModel, 0, "treeRoot"));
    
    int *pPoints1 = (int *) mxGetData(mxGetField(pModel, 0, "pixel1"));
    int *pPoints2 = (int *) mxGetData(mxGetField(pModel, 0, "pixel2"));
    
    const unsigned char* ppCutpoint[2];
    ppCutpoint[0] = (unsigned char *) mxGetData(mxGetField(pModel, 0, "cutpoint"));
    ppCutpoint[1] = ppCutpoint[0] + numBranchNodes;
    
    const int *pLeftChild = (int *) mxGetData(mxGetField(pModel, 0, "leftChild"));
    const int *pRightChild = (int *) mxGetData(mxGetField(pModel, 0, "rightChild"));
    const float *pFit = (float *) mxGetData(mxGetField(pModel, 0, "fit"));
    
    vector<unsigned char *> ppNpdTable(256);
    ppNpdTable[0] = (unsigned char *) mxGetData(mxGetField(pModel, 0, "npdTable"));
    for(int i = 1; i < 256; i++) ppNpdTable[i] = ppNpdTable[i-1] + 256;
    
    /* Validate the input data type */
    if(!mxIsUint8(prhs[1]))
    {
        mexErrMsgTxt("The data type of the second input array must be uint8.");
    }    
    
    size_t height = mxGetDimensions(prhs[1])[0];
    size_t width = mxGetDimensions(prhs[1])[1];
    
    if(height != objSize || width != objSize)
    {
        printf("height=%d, width=%d, objSize=%d\n", height, width, objSize);
        mexErrMsgTxt("The patch dimensions disagree.");
    }
    
    size_t numPixels = objSize * objSize;
    size_t numRects = 1;
    
    if(mxGetNumberOfDimensions(prhs[1]) > 2) numRects = mxGetDimensions(prhs[1])[2];
    
    size_t numIndex = 0;
    int *pIndex;
    
    if(nrhs >= 3 && !mxIsEmpty(prhs[2]))
    {
        /* Validate the input data type */
        if(!mxIsInt32(prhs[2]))
        {
            mexErrMsgTxt("The data types of the third input array must be int32.");
        }
        
        numIndex = mxGetNumberOfElements(prhs[2]);
        pIndex = (int*) mxGetData(prhs[2]);
    }
    
    if(numIndex > numRects) mexErrMsgTxt("Error index.");
    
    size_t numRun = numRects;
    if(numIndex > 0) numRun = numIndex;
    
    if(numRects >= pow(2.0,31)) mexErrMsgTxt("The number of samples is out of the int32 range (required by OpenMP).");
    
    const unsigned char *pPatches = (unsigned char *) mxGetData(prhs[1]);
    
    /* Create mxArrays for the output data and initialize to 0 */
    plhs[0] = mxCreateDoubleMatrix(numRun, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(numRun, 1, mxREAL);
    
    /* Create pointers to the output data */
    double *score = (double*) mxGetData(plhs[0]);
    double *passCount = (double*) mxGetData(plhs[1]);
        
    #pragma omp parallel for

    // process each subwindow
    for(int i = 0; i < numRun; i++) // slide in column
    {
        const unsigned char *pPixel;
        if(numIndex > 0)
        {
            if(pIndex[i] > numRects) mexErrMsgTxt("Error index.");
            pPixel = pPatches + numPixels * size_t(pIndex[i] - 1);
        }        
        else pPixel = pPatches + numPixels * size_t(i);
        
        int treeIndex = 0;
        float Fx = 0;
        int count = 0;

        // test each tree classifier
        for(int s = 0; s < numStages; s++) 
        {
            int node = pTreeRoot[treeIndex];

            // test the current tree classifier
            while(node > -1) // branch node
            {
                unsigned char p1 = pPixel[pPoints1[node]];
                unsigned char p2 = pPixel[pPoints2[node]];
                unsigned char fea = ppNpdTable[p1][p2];

                if(fea < ppCutpoint[0][node] || fea > ppCutpoint[1][node]) node = pLeftChild[node];
                else node = pRightChild[node];
            }

            // leaf node
            node = - node - 1;
            Fx = Fx + pFit[node];
            treeIndex++;

            if(Fx < pStageThreshold[s]) break; // negative samples
            else count++;
        }        

        score[i] = Fx;
        passCount[i] = count;
    }
}
