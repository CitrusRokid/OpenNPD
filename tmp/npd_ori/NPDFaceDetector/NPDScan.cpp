/* Mex function for detecting faces by the normalized pixel difference (NPD) feature based face detector
 *
 * rects = NPDScan(model, I [, minFace, maxFace, numThreads])
 * 
 * Input:
 * 
 * model: structure array of the learned detector.
 * I: the original input image.
 * minFace [optional]: minimal size of face to be detected
 * maxFace [optional]: maximal size of face to be detected
 * numThreads [optional]: the maximum number of computing threads. Positive double scalar. Default: the number of all available processors.
 *
 * Output:
 *
 * rects: bounding boxes of the detected faces, as well as the detection score. This is a structure vector containing the following fields:
 *          row: top-left y-coordinate (starting from 1) of the detected face
 *          col: top-left x-coordinate (starting from 1) of the detected face
 *          size: size of the square bounding box
 *          score: score of the detected face
 *
 * Compile:
 *     Windows: mex -largeArrayDims -DWINDOWS 'COMPFLAGS=/openmp $COMPFLAGS' NPDScan.cpp -v
 *     Linux:  mex -largeArrayDims -DLINUX 'CFLAGS=\$CFLAGS -fopenmp' 'LDFLAGS=\$LDFLAGS -fopenmp' NPDScan.cpp -v
 * 
 * Reference:
 *     Shengcai Liao, Anil K. Jain, and Stan Z. Li, "A Fast and Accurate Unconstrained Face Detector," 
 *       IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015 (Accepted).
 *
 * Version: 1.0
 * Date: 2015-03-04
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
    if(nlhs > 1 || nrhs < 2 || nrhs > 5)
    {
        mexErrMsgTxt("Usage: rects = NPDScan(model, I [, minFace, maxFace, numThreads])");
    }
    
    int minFace = 40;
    int maxFace = 3000;
    
    if(nrhs >= 3 && mxGetScalar(prhs[2]) > 0) minFace = (int) mxGetScalar(prhs[2]);
    if(nrhs >= 4 && mxGetScalar(prhs[3]) > 0) maxFace = (int) mxGetScalar(prhs[3]);
    
    // Set the number of threads
    int numProcs = omp_get_num_procs();
    int numThreads = numProcs;
    
    if(nrhs >= 5 && mxGetScalar(prhs[4]) > 0) numThreads = (int) mxGetScalar(prhs[4]);
    
    if(numThreads > numProcs) numThreads = numProcs;
    omp_set_num_threads(numThreads);
    //printf("minFace=%d, maxFace=%d, numThreads=%d\n", minFace, maxFace, numThreads);
    
    // get input pointers
    const mxArray *pModel = prhs[0];
    
    // get the NPD detector
    int objSize = (int) mxGetScalar(mxGetField(pModel, 0, "objSize"));
    int numStages = (int) mxGetScalar(mxGetField(pModel, 0, "numStages"));
    //int numLeafNodes = (int) mxGetScalar(mxGetField(pModel, 0, "numLeafNodes"));
    int numBranchNodes = (int) mxGetScalar(mxGetField(pModel, 0, "numBranchNodes"));
    const float *pStageThreshold = (float *) mxGetData(mxGetField(pModel, 0, "stageThreshold"));
    const int *pTreeRoot = (int *) mxGetData(mxGetField(pModel, 0, "treeRoot"));
    
    int numScales = (int) mxGetScalar(mxGetField(pModel, 0, "numScales"));
    vector<int *> ppPoints1(numScales);
    vector<int *> ppPoints2(numScales);
    ppPoints1[0] = (int *) mxGetData(mxGetField(pModel, 0, "pixel1"));
    ppPoints2[0] = (int *) mxGetData(mxGetField(pModel, 0, "pixel2"));
    for(int i = 1; i < numScales; i++)
    {
        ppPoints1[i] = ppPoints1[i-1] + numBranchNodes;
        ppPoints2[i] = ppPoints2[i-1] + numBranchNodes;
    }
    
    const unsigned char* ppCutpoint[2];
    ppCutpoint[0] = (unsigned char *) mxGetData(mxGetField(pModel, 0, "cutpoint"));
    ppCutpoint[1] = ppCutpoint[0] + numBranchNodes;
    
    const int *pLeftChild = (int *) mxGetData(mxGetField(pModel, 0, "leftChild"));
    const int *pRightChild = (int *) mxGetData(mxGetField(pModel, 0, "rightChild"));
    const float *pFit = (float *) mxGetData(mxGetField(pModel, 0, "fit"));
    
    vector<unsigned char *> ppNpdTable(256);
    ppNpdTable[0] = (unsigned char *) mxGetData(mxGetField(pModel, 0, "npdTable"));
    for(int i = 1; i < 256; i++) ppNpdTable[i] = ppNpdTable[i-1] + 256;    
    
    //double scaleFactor = mxGetScalar(mxGetField(pModel, 0, "scaleFactor"));
    const int *pWinSize = (int *) mxGetData(mxGetField(pModel, 0, "winSize"));
    
    int height = (int) mxGetM(prhs[1]);
    int width = (int) mxGetN(prhs[1]);
    const unsigned char *I = (unsigned char *) mxGetData(prhs[1]);
    
    minFace = max(minFace, objSize);
    maxFace = min(maxFace, min(height, width));
    
    if(min(height, width) < minFace)
    {
        // create a structure vector for the output data
        const char* field_names[] = {"row", "col", "size", "score"};
        plhs[0] = mxCreateStructMatrix(0, 1, 4, field_names);
        return;
    }
    
    // containers for the detected faces
    vector<double> row, col, size, score;    
    
    for(int k = 0; k < numScales; k++) // process each scale
    {
        if(pWinSize[k] < minFace) continue;
        else if(pWinSize[k] > maxFace) break;
        
        // determine the step of the sliding subwindow
        int winStep = (int) floor(pWinSize[k] * 0.1);
        if(pWinSize[k] > 40) winStep = (int) floor(pWinSize[k] * 0.05);
        
        // calculate the offset values of each pixel in a subwindow
        // pre-determined offset of pixels in a subwindow
        vector<int> offset(pWinSize[k] * pWinSize[k]);
        int p1 = 0, p2 = 0, gap = height - pWinSize[k];
        
        for(int j=0; j < pWinSize[k]; j++) // column coordinate
        {
            for(int i = 0; i < pWinSize[k]; i++) // row coordinate
            {
                offset[p1++] = p2++;
            }
            
            p2 += gap;
        }
        
        int colMax = width - pWinSize[k] + 1;
        int rowMax = height - pWinSize[k] + 1;
        
        #pragma omp parallel for //private(c, pPixel, r, treeIndex, _score, s, node, p1, p2, fea, _row, _col, _size)

        // process each subwindow
        for(int c = 0; c < colMax; c += winStep) // slide in column
        {
            const unsigned char *pPixel = I + c * height;
            
            for(int r = 0; r < rowMax; r += winStep, pPixel += winStep) // slide in row
            {
                int treeIndex = 0;
                float _score = 0;
                int s;

                // test each tree classifier
                for(s = 0; s < numStages; s++) 
                {
                    int node = pTreeRoot[treeIndex];

                    // test the current tree classifier
                    while(node > -1) // branch node
                    {
                        unsigned char p1 = pPixel[offset[ppPoints1[k][node]]];
                        unsigned char p2 = pPixel[offset[ppPoints2[k][node]]];
                        unsigned char fea = ppNpdTable[p1][p2];
                    //printf("node = %d, fea = %d, cutpoint = (%d, %d)\n", node, int(fea), int(ppCutpoint[0][node]), int(ppCutpoint[1][node]));

                        if(fea < ppCutpoint[0][node] || fea > ppCutpoint[1][node]) node = pLeftChild[node];
                        else node = pRightChild[node];
                    }

                    // leaf node
                    node = - node - 1;
                    _score = _score + pFit[node];
                    treeIndex++;

                    //printf("stage = %d, score = %f\n", s, _score);
                    if(_score < pStageThreshold[s]) break; // negative samples
                }
                
                if(s == numStages) // a face detected
                {
                    double _row = r + 1;
                    double _col = c + 1;
                    double _size = pWinSize[k];                    
                    
                    #pragma omp critical // modify the record by a single thread
                    {
                        row.push_back(_row);
                        col.push_back(_col);
                        size.push_back(_size);
                        score.push_back(_score);
                    }
                }
            }
        }
    }
    
    int numFaces = (int) row.size();
    
    // create a structure vector for the output data
    const char* field_names[] = {"row", "col", "size", "score"};
    plhs[0] = mxCreateStructMatrix(numFaces, 1, 4, field_names);
    
    if(numFaces == 0) return;
    
    mxArray *temp;
    
    // asign the output data
    for(int i = 0; i < numFaces; i++)
    {
        temp = mxCreateDoubleMatrix(1,1,mxREAL);
        *mxGetPr(temp) = row[i];
        mxSetField(plhs[0], i, "row", temp);
        
        temp = mxCreateDoubleMatrix(1,1,mxREAL);
        *mxGetPr(temp) = col[i];
        mxSetField(plhs[0], i, "col", temp);
        
        temp = mxCreateDoubleMatrix(1,1,mxREAL);
        *mxGetPr(temp) = size[i];
        mxSetField(plhs[0], i, "size", temp);
        
        temp = mxCreateDoubleMatrix(1,1,mxREAL);
        *mxGetPr(temp) = score[i];
        mxSetField(plhs[0], i, "score", temp);
    }
}
